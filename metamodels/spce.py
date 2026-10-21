"""Stochastic polynomial chaos expansion with MLE coefficient uncertainty.

The SPCE models the conditional response distribution through a polynomial
expansion in the physical inputs and one uniform latent variable. A Gaussian
regularization term makes its conditional density a finite mixture.
"""
import math

import numpy as np
from numpy.polynomial.hermite_e import hermevander
from numpy.polynomial.legendre import legvander
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr

from .base import ArrayModel, integer, positive
from .contracts import CompatibilityError, Prediction
from .pce import total_degree_indices


class StochasticPCE(ArrayModel):
    capabilities = frozenset({
        "mean", "epistemic_variance", "mean_gradient",
        "epistemic_variance_gradient",
    })

    def __init__(self, cfg):
        super().__init__(cfg)
        self.degree = integer(cfg.get("degree", 2), "degree", 1)
        self.max_terms = integer(cfg.get("max_terms", 2000), "max_terms")
        self.likelihood_quadrature_points = integer(
            cfg.get("likelihood_quadrature_points", 32),
            "likelihood_quadrature_points", 4)
        self.failure_quadrature_points = integer(
            cfg.get("failure_quadrature_points", 100),
            "failure_quadrature_points", 4)
        self.coefficient_draws_count = integer(
            cfg.get("coefficient_draws", 100), "coefficient_draws", 2)
        self.optimizer_maxiter = integer(
            cfg.get("optimizer_maxiter", 500), "optimizer_maxiter")
        self.noise_candidates = integer(
            cfg.get("noise_candidates", 5), "noise_candidates", 2)
        self.information_floor = positive(
            cfg.get("information_floor", 1e-8), "information_floor")
        self.initial_noise_fraction = positive(
            cfg.get("initial_noise_fraction", 0.2), "initial_noise_fraction")
        self.noise_damping_fraction = positive(
            cfg.get("noise_damping_fraction", 0.05),
            "noise_damping_fraction", allow_zero=True)
        self._fit_count = 0
        self._warm_parameters = None
        self._initialize_quadrature()

    def _initialize_quadrature(self):
        from numpy.polynomial.legendre import leggauss

        nodes, weights = leggauss(self.likelihood_quadrature_points)
        self.likelihood_nodes = (nodes + 1.0) / 2.0
        self.likelihood_weights = weights / 2.0
        nodes, weights = leggauss(self.failure_quadrature_points)
        self.failure_nodes = (nodes + 1.0) / 2.0
        self.failure_weights = weights / 2.0

    def _check_size(self, dimension):
        count = math.comb(dimension + 1 + self.degree, self.degree)
        if count > self.max_terms:
            raise ValueError(
                f"SPCE needs {count} terms, above max_terms={self.max_terms}")

    def validate_problem(self, problem):
        if getattr(problem, "input_distribution", None) != "standard_normal_independent":
            raise CompatibilityError(
                "AL-SPCE currently requires independent standard-normal physical inputs")
        if not getattr(problem, "stochastic_response", False):
            raise CompatibilityError(
                "AL-SPCE requires a stochastic response simulator")
        self._check_size(problem.d)

    def _basis(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension + 1:
            raise ValueError("SPCE basis requires physical inputs plus one latent coordinate")
        tables = [
            hermevander(points[:, j], self.degree) / self.norm
            for j in range(self.dimension)
        ]
        tables.append(
            legvander(2.0 * points[:, -1] - 1.0, self.degree)
            * self.uniform_norm)
        basis = np.ones((len(points), len(self.indices)))
        for j, table in enumerate(tables):
            basis *= table[:, self.indices[:, j]]
        return basis

    def _basis_and_gradient(self, x, nodes):
        points = np.column_stack([
            np.repeat(np.asarray(x, dtype=float)[None, :], len(nodes), axis=0),
            nodes,
        ])
        tables = [
            hermevander(points[:, j], self.degree) / self.norm
            for j in range(self.dimension)
        ]
        tables.append(
            legvander(2.0 * points[:, -1] - 1.0, self.degree)
            * self.uniform_norm)
        basis = np.ones((len(points), len(self.indices)))
        for j, table in enumerate(tables):
            basis *= table[:, self.indices[:, j]]
        derivatives = np.zeros((len(points), len(self.indices), self.dimension))
        for j in range(self.dimension):
            degrees = self.indices[:, j]
            derivative_table = np.zeros_like(tables[j])
            if self.degree:
                derivative_table[:, 1:] = (
                    tables[j][:, :-1] * np.sqrt(np.arange(1, self.degree + 1)))
            part = derivative_table[:, degrees]
            for k, table in enumerate(tables):
                if k != j:
                    part = part * table[:, self.indices[:, k]]
            derivatives[:, :, j] = part
        return basis, derivatives

    def _failure_basis(self, X):
        points = np.asarray(X, dtype=float)
        physical = np.repeat(points, len(self.failure_nodes), axis=0)
        latent = np.tile(self.failure_nodes, len(points))[:, None]
        return self._basis(np.column_stack((physical, latent))).reshape(
            len(points), len(self.failure_nodes), len(self.indices))

    def _training_features(self, x):
        n = len(x)
        physical = np.repeat(x, len(self.likelihood_nodes), axis=0)
        latent = np.tile(self.likelihood_nodes, n)[:, None]
        return self._basis(np.column_stack((physical, latent))).reshape(
            n, len(self.likelihood_nodes), len(self.indices))

    def _negative_log_likelihood_and_gradient(self, parameters, features, y):
        count = len(self.indices)
        coefficients = parameters[:count]
        sigma = math.exp(float(parameters[count]))
        prediction = np.einsum("nqt,t->nq", features, coefficients)
        residual = (y[:, None] - prediction) / sigma
        log_components = (
            np.log(self.likelihood_weights)[None, :]
            - math.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * residual**2
        )
        log_density = logsumexp(log_components, axis=1)
        posterior_weights = np.exp(log_components - log_density[:, None])
        grad_coefficients = -np.einsum(
            "nq,nq,nqt->t",
            posterior_weights,
            residual / sigma,
            features,
        ) / len(y)
        grad_log_sigma = -float(np.mean(
            np.sum(posterior_weights * (residual**2 - 1.0), axis=1)))
        return (-float(np.mean(log_density)),
                np.concatenate((grad_coefficients, [grad_log_sigma])))

    def _coefficient_covariance(self, coefficients, log_sigma, features, y):
        parameter_count = len(coefficients)
        hessian = np.empty((parameter_count, parameter_count))
        step_scale = np.cbrt(np.finfo(float).eps)
        for j in range(parameter_count):
            step = step_scale * max(1.0, abs(float(coefficients[j])))
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[j] += step
            minus[j] -= step
            p = np.append(plus, log_sigma)
            m = np.append(minus, log_sigma)
            gradient_plus = self._negative_log_likelihood_and_gradient(p, features, y)[1]
            gradient_minus = self._negative_log_likelihood_and_gradient(m, features, y)[1]
            hessian[:, j] = (
                gradient_plus[:parameter_count] -
                gradient_minus[:parameter_count]) / (2.0 * step)
        information = (hessian + hessian.T) * (0.5 * len(y))
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        eigenvalues = np.maximum(eigenvalues, self.information_floor)
        return (eigenvectors / eigenvalues) @ eigenvectors.T

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        self._check_size(x.shape[1])
        self.indices = np.asarray(list(total_degree_indices(
            self.dimension + 1, self.degree)), dtype=int)
        self.norm = np.sqrt([
            float(math.factorial(i)) for i in range(self.degree + 1)])
        self.uniform_norm = np.sqrt(2.0 * np.arange(self.degree + 1) + 1.0)
        features = self._training_features(x)
        self.y_center = float(y.mean())
        self.y_scale = max(float(y.std()), 1e-8)
        scaled_y = (y - self.y_center) / self.y_scale
        mean_features = np.einsum(
            "q,nqt->nt", self.likelihood_weights, features)
        initial_coefficients = np.linalg.lstsq(
            mean_features, scaled_y, rcond=None)[0]
        residual_std = max(float(np.std(scaled_y - mean_features @ initial_coefficients)), 1e-3)
        latent_linear = np.flatnonzero(
            np.all(self.indices[:, :self.dimension] == 0, axis=1) &
            (self.indices[:, -1] == 1))
        if len(latent_linear):
            initial_coefficients[latent_linear[0]] = residual_std
        initial_sigma = max(
            self.initial_noise_fraction * residual_std, 1e-3)
        if self._warm_parameters is not None:
            initial_parameters = self._warm_parameters.copy()
            log_sigma = float(initial_parameters[-1])
            damping = self.noise_damping_fraction
            lower = max(1e-6, math.exp(log_sigma) * (1.0 - damping))
            upper = math.exp(log_sigma) * (1.0 + damping)
            candidate_logs = np.linspace(
                math.log(lower), math.log(upper), self.noise_candidates)
            best = None
            for candidate_log_sigma in candidate_logs:
                def fixed_noise_objective(coefficients):
                    parameters = np.append(coefficients, candidate_log_sigma)
                    loss, gradient = self._negative_log_likelihood_and_gradient(
                        parameters, features, scaled_y)
                    return loss, gradient[:-1]

                candidate = minimize(
                    fixed_noise_objective,
                    self._warm_parameters[:len(self.indices)],
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": self.optimizer_maxiter, "ftol": 1e-10},
                )
                if candidate.success and np.isfinite(candidate.fun) and (
                        best is None or candidate.fun < best.fun):
                    best = candidate
                    selected_log_sigma = float(candidate_log_sigma)
            if best is None:
                raise RuntimeError("SPCE maximum-likelihood fit failed for all damped noise candidates")
            coefficients = np.asarray(best.x, dtype=float)
            parameters = np.append(coefficients, selected_log_sigma)
            result_fun = float(best.fun)
            result_nit = int(best.nit)
        else:
            initial_parameters = np.append(
                initial_coefficients, math.log(initial_sigma))
            result = minimize(
                self._negative_log_likelihood_and_gradient,
                initial_parameters,
                args=(features, scaled_y),
                jac=True,
                method="L-BFGS-B",
                bounds=[(None, None)] * len(self.indices) +
                       [(math.log(1e-6), math.log(10.0))],
                options={"maxiter": self.optimizer_maxiter, "ftol": 1e-10},
            )
            if not result.success or not np.isfinite(result.fun):
                raise RuntimeError(
                    f"SPCE maximum-likelihood fit failed: {result.message}")
            parameters = np.asarray(result.x, dtype=float)
            coefficients = parameters[:len(self.indices)]
            result_fun = float(result.fun)
            result_nit = int(result.nit)

        self.coefficients = coefficients
        self.noise_std = math.exp(float(parameters[-1]))
        self.coefficient_covariance = self._coefficient_covariance(
            self.coefficients, float(parameters[-1]), features, scaled_y)
        rng = np.random.default_rng(self.seed + self._fit_count)
        self.coefficient_draws = rng.multivariate_normal(
            self.coefficients, self.coefficient_covariance,
            size=self.coefficient_draws_count)
        self.coefficient_draws[0] = self.coefficients
        self._warm_parameters = parameters.copy()
        self._fit_count += 1
        self._finish(
            [self.indices, self.coefficients, self.coefficient_covariance,
             self.coefficient_draws, self.noise_std],
            n_terms=len(self.indices),
            design_rank=int(np.linalg.matrix_rank(mean_features)),
            optimizer_iterations=result_nit,
            negative_log_likelihood=result_fun,
            noise_damping_fraction=self.noise_damping_fraction,
            noise_candidates=self.noise_candidates,
            noise_std_standardized_y=self.noise_std,
            likelihood_quadrature_points=self.likelihood_quadrature_points,
            failure_quadrature_points=self.failure_quadrature_points,
            coefficient_draws=self.coefficient_draws_count,
            coefficient_covariance="inverse_observed_information_regularized",
        )

    def input_pdf(self, X):
        points = np.asarray(X, dtype=float)
        return np.exp(-0.5 * np.sum(points**2, axis=1)) / (
            2.0 * math.pi) ** (self.dimension / 2.0)

    def conditional_failure_draws(self, x, gradient=False):
        self._input(x)
        basis, derivatives = self._basis_and_gradient(x, self.failure_nodes)
        means = basis @ self.coefficient_draws.T
        threshold = -self.y_center / self.y_scale
        probabilities = ndtr((threshold - means) / self.noise_std)
        draws = self.failure_weights @ probabilities
        if not gradient:
            return draws
        mean_gradients = np.einsum(
            "qtd,mt->qmd", derivatives, self.coefficient_draws)
        standardized = (threshold - means) / self.noise_std
        density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
        probability_gradients = (
            -density[:, :, None] * mean_gradients / self.noise_std)
        gradients = np.einsum(
            "q,qmd->md", self.failure_weights, probability_gradients)
        return draws, gradients

    def conditional_failure_probability(self, x):
        self._input(x)
        basis, _ = self._basis_and_gradient(x, self.failure_nodes)
        means = basis @ self.coefficients
        threshold = -self.y_center / self.y_scale
        return float(self.failure_weights @
                     ndtr((threshold - means) / self.noise_std))

    def conditional_failure_draws_many(self, X):
        points = np.asarray(X, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("Expected a finite matrix of physical inputs")
        output = []
        threshold = -self.y_center / self.y_scale
        for start in range(0, len(points), 32):
            basis = self._failure_basis(points[start:start + 32])
            means = np.einsum("nqt,mt->nqm", basis, self.coefficient_draws)
            probabilities = ndtr((threshold-means)/self.noise_std)
            output.append(np.einsum(
                "q,nqm->nm", self.failure_weights, probabilities))
        return np.concatenate(output, axis=0) if output else np.empty(
            (0, self.coefficient_draws_count))

    def conditional_failure_probability_many(self, X):
        points = np.asarray(X, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("Expected a finite matrix of physical inputs")
        output = []
        threshold = -self.y_center / self.y_scale
        for start in range(0, len(points), 256):
            basis = self._failure_basis(points[start:start + 256])
            means = np.einsum("nqt,t->nq", basis, self.coefficients)
            probabilities = ndtr((threshold-means)/self.noise_std)
            output.append(probabilities @ self.failure_weights)
        return np.concatenate(output) if output else np.empty(0)

    def predict(self, x, gradient=False):
        self._input(x)
        basis, derivatives = self._basis_and_gradient(x, self.failure_nodes)
        mean_draws = self.failure_weights @ (basis @ self.coefficient_draws.T)
        mean = self.y_center + self.y_scale * float(
            self.failure_weights @ (basis @ self.coefficients))
        variance = float(np.var(mean_draws, ddof=1) * self.y_scale**2)
        mean_gradient = variance_gradient = None
        if gradient:
            mean_draw_gradients = np.einsum(
                "q,qtd,t->d", self.failure_weights, derivatives,
                self.coefficients)
            draw_gradients = np.einsum(
                "q,qtd,mt->md", self.failure_weights, derivatives,
                self.coefficient_draws)
            mean_gradient = tuple(self.y_scale * mean_draw_gradients)
            variance_gradient = tuple(
                2.0 * self.y_scale**2 *
                np.sum((mean_draws - mean_draws.mean())[:, None] *
                       draw_gradients, axis=0) /
                (len(mean_draws) - 1))
        return Prediction(
            mean=mean,
            epistemic_variance=max(variance, 0.0),
            mean_gradient=mean_gradient,
            epistemic_variance_gradient=variance_gradient,
        )
