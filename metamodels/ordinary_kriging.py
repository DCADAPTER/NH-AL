"""Constant-trend Gaussian ordinary Kriging: GLS, profile MLE and BLUP MSE.

P01 Section 2, Eqs.(6)-(11); numerical scaling follows the DACE manual (https://omicron.dk/dace.html).
The optimizer is SciPy L-BFGS-B, not a port of DACE's boxmin algorithm.
"""
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from .base import ArrayModel, integer, positive
from .contracts import Prediction


class OrdinaryKriging(ArrayModel):
    capabilities = frozenset({'mean', 'epistemic_variance', 'input_gradient',
                              'mean_gradient', 'epistemic_variance_gradient'})

    def __init__(self, cfg):
        super().__init__(cfg)
        self.theta0 = np.atleast_1d(np.asarray(cfg.get('theta', 1.0), dtype=float))
        if self.theta0.ndim != 1 or not len(self.theta0) or not np.isfinite(self.theta0).all() or (self.theta0 <= 0).any():
            raise ValueError('theta must be a positive scalar or vector')
        self.bounds = np.asarray(cfg.get('theta_bounds', [1e-3, 1e3]), dtype=float)
        if self.bounds.shape != (2,) or not np.isfinite(self.bounds).all() or not 0 < self.bounds[0] < self.bounds[1]:
            raise ValueError('theta_bounds must be [positive lower, larger upper]')
        if ((self.theta0 < self.bounds[0]) | (self.theta0 > self.bounds[1])).any():
            raise ValueError('theta must lie within theta_bounds')
        self.nugget = positive(cfg.get('nugget', 1e-12), 'nugget', allow_zero=True)
        self.optimize = cfg.get('optimize', True)
        if not isinstance(self.optimize, bool):
            raise ValueError('optimize must be boolean')
        self.restarts = integer(cfg.get('optimizer_restarts', 2), 'optimizer_restarts', 0)
        self.batch_size = integer(cfg.get('batch_size', 4096), 'batch_size')

    def validate_problem(self, problem):
        if len(self.theta0) not in (1, problem.d):
            raise ValueError('theta vector must match problem dimension')

    def _state(self, log_theta):
        theta = np.exp(log_theta)
        r = np.exp(-cdist(self.z * np.sqrt(theta), self.z * np.sqrt(theta), 'sqeuclidean'))
        r.flat[::len(r)+1] += self.nugget
        factor = cho_factor(r, lower=True, check_finite=False)
        ri_one = cho_solve(factor, np.ones(len(r)), check_finite=False)
        denominator = ri_one.sum()
        beta = (ri_one @ self.t) / denominator
        residual = self.t - beta
        alpha = cho_solve(factor, residual, check_finite=False)
        sigma2 = max(float(residual @ alpha / len(r)), np.finfo(float).tiny)
        objective = len(r) * np.log(sigma2) + 2 * np.log(np.diag(factor[0])).sum()
        return float(objective), (theta, factor, ri_one, denominator, float(beta), alpha, sigma2)

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        if len(self.theta0) not in (1, self.dimension):
            raise ValueError('theta vector must match problem dimension')
        # DACE convention: sample standard deviations (ddof=1).
        self.x_center = x.mean(axis=0)
        self.x_scale = x.std(axis=0, ddof=1)
        self.x_scale[self.x_scale < 1e-12] = 1.0
        self.y_center = float(y.mean())
        self.y_scale = max(float(y.std(ddof=1)), 1e-8)
        self.z = (x - self.x_center) / self.x_scale
        self.t = (y - self.y_center) / self.y_scale
        initial = np.log(np.broadcast_to(self.theta0, (self.dimension,)))
        best_objective, best = self._state(initial)
        messages = []
        def objective(v):
            try:
                return self._state(v)[0]
            except np.linalg.LinAlgError:
                return 1e100
        if self.optimize:
            rng = np.random.default_rng(self.seed)
            lo, hi = np.log(self.bounds)
            starts = [initial] + [rng.uniform(lo, hi, self.dimension) for _ in range(self.restarts)]
            for start in starts:
                result = minimize(objective, start, method='L-BFGS-B', bounds=[(lo, hi)] * self.dimension)
                if not result.success:
                    messages.append(str(result.message))
                value = objective(result.x)
                if value < best_objective:
                    best_objective, best = self._state(result.x)
        self.theta, self.factor, self.ri_one, self.denominator, self.beta, self.alpha, self.sigma2 = best
        for message in messages:
            warnings.warn('OrdinaryKriging optimizer: ' + message, RuntimeWarning, stacklevel=2)
        self._finish([self.theta, self.beta, self.sigma2, self.alpha],
                     trend='constant_GLS', correlation='exp(-sum(theta_j*delta_j**2))',
                     theta=self.theta.tolist(), beta=self.y_center+self.y_scale*self.beta,
                     process_variance=self.sigma2*self.y_scale**2,
                     estimation='profile_maximum_likelihood', objective=best_objective,
                     optimizer='L-BFGS-B' if self.optimize else 'fixed_theta',
                     input_scaling='training_mean_sample_std', nugget=self.nugget,
                     warnings=messages, variance_kind='ordinary_kriging_MSE_including_trend_uncertainty')

    def _moments(self, z):
        r = np.exp(-cdist(z*np.sqrt(self.theta), self.z*np.sqrt(self.theta), 'sqeuclidean'))
        solved = cho_solve(self.factor, r.T, check_finite=False).T
        correction = 1 - r @ self.ri_one
        mu = self.y_center + self.y_scale*(self.beta + r @ self.alpha)
        raw = self.y_scale**2*self.sigma2*(1 - np.sum(r*solved, axis=1) + correction**2/self.denominator)
        return mu, raw, r, solved, correction

    def predict_moments(self, X):
        if self.snapshot is None:
            raise RuntimeError('Model is not fitted; call fit first')
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('Expected finite X with shape (n,d)')
        means, variances = np.empty(len(x)), np.empty(len(x))
        for start in range(0, len(x), self.batch_size):
            end = start + self.batch_size
            mu, raw, *_ = self._moments((x[start:end]-self.x_center)/self.x_scale)
            means[start:end], variances[start:end] = mu, np.maximum(raw, 1e-12)
        return means, variances

    def predict(self, x, gradient=False):
        z = (self._input(x)-self.x_center)/self.x_scale
        mu, raw, r, solved, correction = self._moments(z[None, :])
        dm = dv = None
        if gradient:
            dr = -2*self.theta*(z-self.z)/self.x_scale*r[0, :, None]
            dm = tuple(self.y_scale*(dr.T @ self.alpha))
            dv = tuple(self.y_scale**2*self.sigma2*(-2*dr.T @ solved[0]
                       - 2*correction[0]/self.denominator*(dr.T @ self.ri_one))) if raw[0] > 1e-12 else (0.,)*self.dimension
        return Prediction(mean=float(mu[0]), epistemic_variance=max(float(raw[0]), 1e-12),
                          mean_gradient=dm, epistemic_variance_gradient=dv)
