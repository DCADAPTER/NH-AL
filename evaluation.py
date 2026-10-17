"""Surrogate Pf integration and explicit stopping contracts; no true g calls."""
import math
import importlib
import time
from metamodels.contracts import CompatibilityError


def _mc_metrics(model, integration):
    field = "mean" if "mean" in model.capabilities else "decision_function"
    if field == "mean" and callable(getattr(model, "predict_moments", None)):
        mean, _ = model.predict_moments(integration)
        failures = int((mean <= 0).sum())
    else:
        failures = sum(model.predict(x).require(field) <= 0 for x in integration)
    n = len(integration)
    pf = failures / n
    se = math.sqrt(pf * (1 - pf) / n)
    return {"pf_surrogate": pf, "mc_se_conditional_on_model": se,
            "mc_cov": se / pf if failures > 0 else None}


class BudgetMCEvaluation:
    """Plug-in sign estimate; stop only at budget, with no convergence claim."""
    required_capabilities = frozenset()
    required_any_capabilities = frozenset({"mean", "decision_function"})

    def __init__(self, cfg, integration, audit):
        self.cfg = cfg
        self.integration = integration
        self.audit = audit
        if int(cfg["max_true_calls"]) < 2:
            raise ValueError("max_true_calls must be >= 2")

    def evaluate(self, model, acquisition, n_calls):
        start = time.perf_counter()
        metrics = _mc_metrics(model, self.integration)
        budget = n_calls >= self.cfg["max_true_calls"]
        return {**metrics, "converged": False, "stop": budget,
                "stop_reason": "budget_exhausted" if budget else "continue",
                "evaluation_seconds": time.perf_counter() - start}


class MCEvaluation(BudgetMCEvaluation):
    """U audit heuristic, explicitly restricted to U acquisition."""
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    supported_acquisition_kinds = frozenset({"u"})

    def __init__(self, cfg, integration, audit):
        super().__init__(cfg, integration, audit)
        for name in ("max_mc_cov", "min_audit_u"):
            if not math.isfinite(cfg[name]) or cfg[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def evaluate(self, model, acquisition, n_calls):
        if getattr(acquisition, "kind", None) != "u":
            raise CompatibilityError("MCEvaluation audit threshold requires U acquisition")
        start = time.perf_counter()
        metrics = _mc_metrics(model, self.integration)
        audit_min = min(acquisition.score(x) for x in self.audit)
        cov = metrics["mc_cov"]
        precision = cov is not None and cov <= self.cfg["max_mc_cov"]
        converged = audit_min >= self.cfg["min_audit_u"] and precision
        budget = n_calls >= self.cfg["max_true_calls"]
        reason = "converged_audit_heuristic" if converged else ("budget_exhausted" if budget else "continue")
        return {**metrics, "audit_min_u": audit_min, "converged": converged,
                "stop": converged or budget, "stop_reason": reason,
                "evaluation_seconds": time.perf_counter() - start}


class PluginMCEvaluation(BudgetMCEvaluation):
    """Compute Pf and route method-specific convergence to stop.<name>:Stop.

    A plugin declares capabilities and implements evaluate(model, acquisition,
    population, metrics). It returns stop/converged/stop_reason and optionally
    expand_population. Resource budgets remain the evaluator's responsibility.
    """
    def __init__(self, cfg, integration, audit):
        super().__init__(cfg, integration, audit)
        from metamodels.base import integer
        spec = cfg["stop"]
        module, name = spec["factory"].split(":")
        self.stop_plugin = getattr(importlib.import_module(module), name)(spec)
        if not callable(getattr(self.stop_plugin, "evaluate", None)):
            raise CompatibilityError("Stop plugin must implement evaluate()")
        self.required_capabilities = self.stop_plugin.required_capabilities
        self.supported_acquisition_kinds = self.stop_plugin.supported_acquisition_kinds
        self.requires_shared_population = getattr(self.stop_plugin, "requires_shared_population", False)
        self.population_increment = integer(cfg["population_increment"], "population_increment")
        self.max_population_size = integer(cfg["max_population_size"], "max_population_size")
        if not integration or len(integration) > self.max_population_size:
            raise ValueError("Require 0 < integration_size <= max_population_size")

    def evaluate(self, model, acquisition, n_calls):
        if acquisition.kind not in self.supported_acquisition_kinds:
            raise CompatibilityError("Stop plugin does not support this acquisition")
        start = time.perf_counter()
        metrics = _mc_metrics(model, self.integration)
        decision = self.stop_plugin.evaluate(model, acquisition, self.integration, metrics)
        decision.setdefault("expand_population", False)
        if not decision["converged"]:
            if n_calls >= self.cfg["max_true_calls"]:
                decision.update(stop=True, expand_population=False, stop_reason="budget_exhausted")
            elif decision["expand_population"] and len(self.integration) >= self.max_population_size:
                decision.update(stop=True, expand_population=False, stop_reason="mc_population_limit")
        return {**metrics, **decision, "population_size": len(self.integration),
                "stop_plugin": self.cfg["stop"]["factory"],
                "evaluation_seconds": time.perf_counter() - start}
