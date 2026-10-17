"""Reference selection and acquisition. Never evaluates true g."""
import math

def distance2(x, z):
    return sum((a-b)**2 for a,b in zip(x,z))

class UAcquisition:
    direction = "min"
    kind = "u"
    capabilities = frozenset({"score", "energy", "energy_gradient"})
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    gradient_required_capabilities = frozenset({"mean_gradient", "epistemic_variance_gradient"})
    gradient_kind = "analytic_energy_U_squared"
    def __init__(self, model):
        self.model = model
        self.score_points = 0
        self.energy_points = 0
        self.gradient_points = 0
    def score(self, x):
        self.score_points += 1
        prediction = self.model.predict(x)
        mu, var = prediction.require('mean'), prediction.require('epistemic_variance')
        return abs(mu)/math.sqrt(var)
    def score_many(self, X):
        import numpy as np
        if callable(getattr(self.model, "predict_moments", None)) and type(self).score is UAcquisition.score:
            mu, variance = self.model.predict_moments(X)
            self.score_points += len(X)
            return np.abs(mu)/np.sqrt(variance)
        return np.asarray([self.score(x) for x in X])

    def energy(self, x):
        self.energy_points += 1
        prediction = self.model.predict(x)
        mu, var = prediction.require('mean'), prediction.require('epistemic_variance')
        return mu*mu/var
    def energy_gradient(self, x):
        self.gradient_points += 1
        prediction = self.model.predict(x, gradient=True)
        mu, var = prediction.require('mean'), prediction.require('epistemic_variance')
        dm, dv = prediction.require('mean_gradient'), prediction.require('epistemic_variance_gradient')
        return [2*mu*a/var - mu*mu*b/(var*var) for a,b in zip(dm,dv)]

class MeanBoundaryAcquisition:
    """Minimal |predicted g| baseline; no uncertainty-based convergence claim."""
    direction = "min"
    kind = "mean_boundary"
    capabilities = frozenset({"score", "energy", "energy_gradient"})
    required_capabilities = frozenset({"mean"})
    gradient_required_capabilities = frozenset({"mean_gradient"})
    value_field = "mean"
    gradient_field = "mean_gradient"

    def __init__(self, model):
        self.model = model
        self.score_points = self.energy_points = self.gradient_points = 0

    def score(self, x):
        self.score_points += 1
        return abs(self.model.predict(x).require(self.value_field))

    def energy(self, x):
        self.energy_points += 1
        return self.model.predict(x).require(self.value_field)**2

    def energy_gradient(self, x):
        self.gradient_points += 1
        prediction = self.model.predict(x, gradient=True)
        value = prediction.require(self.value_field)
        return [2*value*v for v in prediction.require(self.gradient_field)]


class SVMMarginAcquisition(MeanBoundaryAcquisition):
    """Unnormalized decision margin baseline, not a paper-specific ASVM policy."""
    kind = "svm_margin"
    required_capabilities = frozenset({"decision_function"})
    gradient_required_capabilities = frozenset({"decision_gradient"})
    value_field = "decision_function"
    gradient_field = "decision_gradient"


class PoolSelector:
    def __init__(self, cfg):
        self.min_distance = float(cfg.get("min_distance",1e-6))
    def valid(self, x, X, problem):
        return problem.admissible(x) and all(distance2(x,z)>self.min_distance**2 for z in X)
    def select(self, candidates, acquisition, X, problem):
        eligible = [x for x in candidates if self.valid(x, X, problem)]
        if not eligible:
            raise RuntimeError("No eligible candidate; not convergence")
        scored = [(acquisition.score(x), i, x) for i,x in enumerate(eligible)]
        best = min(scored) if acquisition.direction=="min" else max(scored,key=lambda v:(v[0],-v[1]))
        return list(best[2]), best[0]

def accept_refinement(x_ref, score_ref, candidates, acquisition, valid):
    """Reference retained on ties; original score, not NH energy, decides."""
    x_best, score_best = list(x_ref), score_ref
    for x in candidates:
        if not valid(x):
            continue
        score = acquisition.score(x)
        better = score < score_best if acquisition.direction=="min" else score > score_best
        if math.isfinite(score) and better:
            x_best, score_best = list(x), score
    return x_best, score_best



class BatchedPoolSelector(PoolSelector):
    """Same min/max and distance policy as PoolSelector, for large MC populations."""
    def select(self, candidates, acquisition, X, problem):
        import numpy as np
        from scipy.spatial import cKDTree
        points = np.asarray(candidates, dtype=float)
        valid = np.asarray([problem.admissible(x) for x in points])
        if len(X):
            distances, _ = cKDTree(X).query(points)
            valid &= distances > self.min_distance
        eligible = points[valid]
        if not len(eligible):
            raise RuntimeError("No eligible candidate; not convergence")
        score_many = getattr(acquisition, "score_many", None)
        scores = score_many(eligible) if callable(score_many) else np.asarray([acquisition.score(x) for x in eligible])
        if not np.isfinite(scores).all():
            raise ValueError("Acquisition scores must be finite")
        index = np.argmin(scores) if acquisition.direction == "min" else np.argmax(scores)
        return eligible[index].tolist(), float(scores[index])
