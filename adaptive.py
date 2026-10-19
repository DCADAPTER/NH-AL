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


class EFFAcquisition:
    """Expected Feasibility Function acquisition from P01, Eq. (14).

    For reliability problems the paper uses limit state zero and epsilon equal
    to twice the Kriging standard deviation. EFF is maximized; NH minimizes
    its negative.
    """
    direction = "max"
    kind = "eff"
    capabilities = frozenset({"score", "score_many", "energy", "energy_gradient"})
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    gradient_required_capabilities = frozenset({"mean_gradient", "epistemic_variance_gradient"})
    gradient_kind = "analytic_energy_negative_eff"

    def __init__(self, model):
        self.model = model
        self.limit_state = 0.0
        self.epsilon_multiplier = 2.0
        self.score_points = 0
        self.energy_points = 0
        self.gradient_points = 0

    def _terms(self, mean, variance):
        import numpy as np
        from scipy.special import ndtr

        sigma = np.sqrt(variance)
        d = (mean-self.limit_state)/sigma
        c = self.epsilon_multiplier
        z0, zlo, zhi = -d, -c-d, c-d
        root_two_pi = math.sqrt(2*math.pi)
        phi0 = np.exp(-.5*z0*z0)/root_two_pi
        philo = np.exp(-.5*zlo*zlo)/root_two_pi
        phihi = np.exp(-.5*zhi*zhi)/root_two_pi
        cdf0, cdflo, cdfhi = ndtr(z0), ndtr(zlo), ndtr(zhi)
        mean_derivative = 2*cdf0-cdflo-cdfhi
        normalized_eff = (c*(cdfhi-cdflo) + d*mean_derivative
                          + philo+phihi-2*phi0)
        eff = np.maximum(sigma*normalized_eff, 0.0)
        sigma_derivative = normalized_eff-d*mean_derivative
        return eff, mean_derivative, sigma_derivative, sigma

    def score(self, x):
        self.score_points += 1
        prediction = self.model.predict(x)
        mean = prediction.require("mean")
        variance = prediction.require("epistemic_variance")
        return float(self._terms(mean, variance)[0])

    def score_many(self, X):
        import numpy as np
        if callable(getattr(self.model, "predict_moments", None)):
            mean, variance = self.model.predict_moments(X)
            self.score_points += len(X)
            return self._terms(mean, variance)[0]
        return np.asarray([self.score(x) for x in X])

    def energy(self, x):
        self.energy_points += 1
        prediction = self.model.predict(x)
        mean = prediction.require("mean")
        variance = prediction.require("epistemic_variance")
        return -float(self._terms(mean, variance)[0])

    def energy_gradient(self, x):
        import numpy as np
        self.gradient_points += 1
        prediction = self.model.predict(x, gradient=True)
        mean = prediction.require("mean")
        variance = prediction.require("epistemic_variance")
        dmean = np.asarray(prediction.require("mean_gradient"), dtype=float)
        dvariance = np.asarray(prediction.require("epistemic_variance_gradient"), dtype=float)
        _, mean_derivative, sigma_derivative, sigma = self._terms(mean, variance)
        gradient = -(mean_derivative*dmean + sigma_derivative*dvariance/(2*sigma))
        return gradient.tolist()


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
