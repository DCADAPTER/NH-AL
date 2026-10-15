"""Surrogate Pf integration and explicit stopping contracts; no true g calls."""
import math
import time
from metamodels.contracts import CompatibilityError


def _mc_metrics(model, integration):
    field = "mean" if "mean" in model.capabilities else "decision_function"
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
