"""P01 AK-MCS-U: population learning check followed by MC precision check.

Thresholds live in YAML. Budget limits are operational, not convergence criteria.
"""
import math
from metamodels.base import positive


class Stop:
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    supported_acquisition_kinds = frozenset({"u"})
    requires_shared_population = True

    def __init__(self, cfg):
        self.population_mode = cfg.get("population_mode", "shared")
        if self.population_mode not in ("shared", "holdout"):
            raise ValueError("stop.population_mode must be shared or holdout")
        self.requires_shared_population = self.population_mode == "shared"
        self.requires_independent_population = self.population_mode == "holdout"
        self.min_u = positive(cfg["min_u"], "stop.min_u")
        self.max_mc_cov = positive(cfg["max_mc_cov"], "stop.max_mc_cov")

    def evaluate(self, model, acquisition, population, metrics):
        if not population:
            raise ValueError("AK-MCS-U requires a nonempty MC population")
        observed = {tuple(x) for x in getattr(model, "X", ())}
        unobserved = (population if self.requires_independent_population else
                      [x for x in population if tuple(x) not in observed])
        minimum = math.inf
        score_many = getattr(acquisition, "score_many", None)
        scores = ()
        if unobserved:
            scores = (score_many(unobserved) if callable(score_many) else
                      (acquisition.score(x) for x in unobserved))
        for value in scores:
            if not math.isfinite(value) or value < 0:
                raise ValueError("U scores must be finite and nonnegative")
            minimum = min(minimum, value)
        learning = bool(minimum >= self.min_u)
        cov = metrics["mc_cov"]
        precision = cov is not None and math.isfinite(cov) and cov <= self.max_mc_cov
        converged = learning and precision
        reason = "converged_holdout_u" if self.requires_independent_population else "converged_ak_mcs_u"
        return {"population_mode": self.population_mode,
                "population_min_u": float(minimum) if unobserved else None,
                "u_scored_points": len(unobserved), "learning_converged": learning,
                "mc_precision_met": precision, "converged": converged,
                "stop": converged,
                "expand_population": learning and not precision,
                "stop_reason": reason if converged else "continue"}
