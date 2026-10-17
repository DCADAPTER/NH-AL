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
        self.min_u = positive(cfg["min_u"], "stop.min_u")
        self.max_mc_cov = positive(cfg["max_mc_cov"], "stop.max_mc_cov")

    def evaluate(self, model, acquisition, population, metrics):
        if not population:
            raise ValueError("AK-MCS-U requires a nonempty MC population")
        minimum = math.inf
        score_many = getattr(acquisition, "score_many", None)
        scores = score_many(population) if callable(score_many) else (acquisition.score(x) for x in population)
        for value in scores:
            if not math.isfinite(value) or value < 0:
                raise ValueError("U scores must be finite and nonnegative")
            minimum = min(minimum, value)
        learning = bool(minimum >= self.min_u)
        cov = metrics["mc_cov"]
        precision = cov is not None and math.isfinite(cov) and cov <= self.max_mc_cov
        converged = learning and precision
        return {"population_min_u": float(minimum), "learning_converged": learning,
                "mc_precision_met": precision, "converged": converged,
                "stop": converged,
                "expand_population": learning and not precision,
                "stop_reason": "converged_ak_mcs_u" if converged else "continue"}
