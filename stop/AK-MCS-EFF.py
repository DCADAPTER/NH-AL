"""P01 AK-MCS-EFF population learning check and MC precision check."""
import math
from metamodels.base import positive


class Stop:
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    supported_acquisition_kinds = frozenset({"eff"})
    requires_shared_population = True

    def __init__(self, cfg):
        self.population_mode = cfg.get("population_mode", "shared")
        if self.population_mode not in ("shared", "holdout"):
            raise ValueError("stop.population_mode must be shared or holdout")
        self.requires_shared_population = self.population_mode == "shared"
        self.requires_independent_population = self.population_mode == "holdout"
        self.max_eff = positive(cfg["max_eff"], "stop.max_eff")
        self.max_mc_cov = positive(cfg["max_mc_cov"], "stop.max_mc_cov")

    def evaluate(self, model, acquisition, population, metrics):
        if not population:
            raise ValueError("AK-MCS-EFF requires a nonempty MC population")
        observed = {tuple(x) for x in getattr(model, "X", ())}
        unobserved = (population if self.requires_independent_population else
                      [x for x in population if tuple(x) not in observed])
        maximum = 0.0
        if unobserved:
            score_many = getattr(acquisition, "score_many", None)
            scores = (score_many(unobserved) if callable(score_many) else
                      (acquisition.score(x) for x in unobserved))
            maximum = -math.inf
            for value in scores:
                if not math.isfinite(value) or value < 0:
                    raise ValueError("EFF scores must be finite and nonnegative")
                maximum = max(maximum, float(value))
        learning = bool(maximum <= self.max_eff)
        cov = metrics["mc_cov"]
        precision = bool(cov is not None and math.isfinite(cov) and
                         cov <= self.max_mc_cov)
        converged = bool(learning and precision)
        reason = ("converged_holdout_eff" if self.requires_independent_population
                  else "converged_ak_mcs_eff")
        return {"population_mode": self.population_mode,
                "population_max_eff": float(maximum) if unobserved else None,
                "eff_scored_points": len(unobserved),
                "learning_converged": learning, "mc_precision_met": precision,
                "converged": converged, "stop": converged,
                "expand_population": bool(learning and not precision),
                "stop_reason": reason if converged else "continue"}
