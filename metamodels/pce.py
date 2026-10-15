"""Total-degree orthonormal Hermite PCE for independent N(0,1) inputs.

Least-squares/ridge coefficients; optional pairs bootstrap is an ensemble
uncertainty proxy, never advertised as a GP posterior variance.
"""
import math
import numpy as np
from numpy.polynomial.hermite_e import hermevander
from .base import ArrayModel, positive, integer
from .contracts import Prediction, CompatibilityError


def total_degree_indices(dimension, degree):
    def exact(d, total):
        if d == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for tail in exact(d - 1, total - first):
                    yield (first,) + tail
    for total in range(degree + 1):
        yield from exact(dimension, total)


class PolynomialChaos(ArrayModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.degree = integer(cfg.get("degree", 3), "degree", 0)
        self.ridge = positive(cfg.get("ridge", 1e-10), "ridge", allow_zero=True)
        self.bootstrap = integer(cfg.get("bootstrap_members", 0), "bootstrap_members", 0)
        if self.bootstrap == 1:
            raise ValueError("bootstrap_members must be 0 or >= 2")
        self.max_terms = integer(cfg.get("max_terms", 2000), "max_terms")
        self.capabilities = frozenset({"mean", "input_gradient", "mean_gradient"} |
            ({"ensemble_responses", "ensemble_variance"} if self.bootstrap else set()))

    def _check_size(self, dimension):
        count = math.comb(dimension + self.degree, self.degree)
        if count > self.max_terms:
            raise ValueError(f"PCE needs {count} terms, above max_terms={self.max_terms}")

    def validate_problem(self, problem):
        if getattr(problem, "input_distribution", None) != "standard_normal_independent":
            raise CompatibilityError("Hermite PCE requires independent standard-normal coordinates")
        self._check_size(problem.d)

    def _basis(self, x, gradient=False):
        # Normalized probabilists' Hermite: He_n / sqrt(n!).
        tables = [hermevander(x[:, j], self.degree) / self.norm
                  for j in range(self.dimension)]
        phi = np.ones((len(x), len(self.indices)))
        for j in range(self.dimension):
            phi *= tables[j][:, self.indices[:, j]]
        if not gradient:
            return phi
        derivatives = np.zeros((len(x), len(self.indices), self.dimension))
        for j in range(self.dimension):
            degree = self.indices[:, j]
            part = tables[j][:, np.maximum(degree - 1, 0)] * np.sqrt(degree)
            for k in range(self.dimension):
                if k != j:
                    part = part * tables[k][:, self.indices[:, k]]
            derivatives[:, :, j] = part
        return phi, derivatives

    def _regress(self, design, y):
        if self.ridge:
            penalty = np.sqrt(self.ridge) * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            design = np.vstack([design, penalty])
            y = np.concatenate([y, np.zeros(len(penalty))])
        return np.linalg.lstsq(design, y, rcond=None)[0]

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        self._check_size(self.dimension)
        self.indices = np.asarray(list(total_degree_indices(self.dimension, self.degree)))
        self.norm = np.sqrt([float(math.factorial(i)) for i in range(self.degree + 1)])
        design = self._basis(x)
        self.coefficients = self._regress(design, y)
        rng = np.random.default_rng(self.seed)
        self.bootstrap_coefficients = []
        for _ in range(self.bootstrap):
            rows = rng.integers(0, len(x), len(x))
            self.bootstrap_coefficients.append(self._regress(design[rows], y[rows]))
        self.bootstrap_coefficients = np.asarray(self.bootstrap_coefficients)
        rank = int(np.linalg.matrix_rank(design))
        self._finish([self.indices, self.coefficients, self.bootstrap_coefficients],
                     n_terms=len(self.indices), design_rank=rank,
                     underdetermined=rank < len(self.indices),
                     basis="orthonormal_probabilists_Hermite_N(0,I)",
                     uncertainty="pairs_bootstrap_proxy" if self.bootstrap else "none")

    def predict(self, x, gradient=False):
        x = self._input(x)[None, :]
        dm = None
        if gradient:
            phi, derivative = self._basis(x, gradient=True)
            dm = tuple(self.coefficients @ derivative[0])
        else:
            phi = self._basis(x)
        samples = tuple(self.bootstrap_coefficients @ phi[0]) if self.bootstrap else None
        return Prediction(mean=float(phi[0] @ self.coefficients), mean_gradient=dm,
                          ensemble_responses=samples,
                          ensemble_variance=float(np.var(samples, ddof=1)) if samples else None)
