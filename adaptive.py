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


class ALSPCEAcquisition:
    """P26 learning function: input density times epistemic Var[s_hat(x)]."""
    direction = "max"
    kind = "al_spce"
    capabilities = frozenset({"score", "score_many", "energy", "energy_gradient"})
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    gradient_required_capabilities = frozenset({
        "mean_gradient", "epistemic_variance_gradient"})
    gradient_kind = "analytic_energy_negative_al_spce"

    def __init__(self, model):
        self.model = model
        self.score_points = self.energy_points = self.gradient_points = 0

    def _score(self, x):
        draws = self.model.conditional_failure_draws(x)
        variance = float(draws.var(ddof=1))
        density = float(self.model.input_pdf([x])[0])
        return density * variance

    def score(self, x):
        self.score_points += 1
        return self._score(x)

    def score_many(self, X):
        import numpy as np
        points = np.asarray(X, dtype=float)
        draws = self.model.conditional_failure_draws_many(points)
        self.score_points += len(points)
        return self.model.input_pdf(points) * draws.var(axis=1, ddof=1)

    def energy(self, x):
        self.energy_points += 1
        return -self._score(x)

    def energy_gradient(self, x):
        import numpy as np
        self.gradient_points += 1
        draws, draw_gradients = self.model.conditional_failure_draws(
            x, gradient=True)
        mean = float(draws.mean())
        variance = float(draws.var(ddof=1))
        variance_gradient = (
            2.0 * np.sum((draws-mean)[:, None] * draw_gradients, axis=0)
            / (len(draws)-1))
        density = float(self.model.input_pdf([x])[0])
        score_gradient = density * (
            variance_gradient - np.asarray(x, dtype=float) * variance)
        return (-score_gradient).tolist()


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


class WeightedKMedoidsBatchSelector(PoolSelector):
    """Weighted K-medoids candidate clustering for AL-SPCE batch enrichment."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.batch_size = int(cfg.get("batch_size", 5))
        self.max_iterations = int(cfg.get("max_iterations", 20))
        if self.batch_size < 1 or self.max_iterations < 1:
            raise ValueError("batch_size and max_iterations must be positive")

    def select(self, candidates, acquisition, X, problem):
        return self.select_batch(candidates, acquisition, X, problem, 1)[0]

    def select_batch(self, candidates, acquisition, X, problem, batch_size=None):
        import numpy as np
        from scipy.spatial.distance import cdist

        points = np.asarray(candidates, dtype=float)
        valid = np.asarray([problem.admissible(x) for x in points])
        if len(X):
            from scipy.spatial import cKDTree
            distances, _ = cKDTree(X).query(points)
            valid &= distances > self.min_distance
        eligible = points[valid]
        if not len(eligible):
            raise RuntimeError("No eligible candidate; not convergence")
        scores = np.asarray(acquisition.score_many(eligible), dtype=float)
        if scores.shape != (len(eligible),) or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Batch acquisition scores must be finite and nonnegative")
        count = min(batch_size or self.batch_size, len(eligible))
        if count == 1:
            index = int(np.argmax(scores) if acquisition.direction == "max"
                        else np.argmin(scores))
            return [(eligible[index].tolist(), float(scores[index]))]

        weights = scores.copy()
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            weights.fill(1.0)
        weights /= weights.sum()
        distances = cdist(eligible, eligible)
        medoids = [int(np.argmax(weights))]
        while len(medoids) < count:
            nearest = np.min(distances[:, medoids], axis=1)
            potential = weights * nearest**2
            potential[medoids] = -1
            next_medoid = int(np.argmax(potential))
            if potential[next_medoid] <= 0:
                next_medoid = next(i for i in range(len(eligible)) if i not in medoids)
            medoids.append(next_medoid)

        for _ in range(self.max_iterations):
            assignments = np.argmin(distances[:, medoids], axis=1)
            updated = list(medoids)
            for cluster in range(count):
                members = np.flatnonzero(assignments == cluster)
                if not len(members):
                    continue
                costs = distances[np.ix_(members, members)] @ weights[members]
                updated[cluster] = int(members[int(np.argmin(costs))])
            if updated == medoids:
                break
            medoids = updated

        assignments = np.argmin(distances[:, medoids], axis=1)
        chosen = []
        for cluster in range(count):
            members = np.flatnonzero(assignments == cluster)
            if not len(members):
                continue
            local = scores[members]
            index = int(members[np.argmax(local) if acquisition.direction == "max"
                                 else np.argmin(local)])
            chosen.append((eligible[index].tolist(), float(scores[index])))
        if len(chosen) != count:
            raise RuntimeError("Weighted K-medoids produced an empty batch cluster")
        return chosen
