"""RBF GP with fitted kernel parameters and analytic input gradients.

Uses sklearn GaussianProcessRegressor; the empirical constant trend is removed
before fitting (not generalized least-squares ordinary Kriging).
"""
import warnings
import numpy as np
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from .base import ArrayModel, positive, integer
from .contracts import Prediction


class KrigingGP(ArrayModel):
    capabilities = frozenset({"mean", "epistemic_variance", "input_gradient",
                              "mean_gradient", "epistemic_variance_gradient"})

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lengthscale = np.asarray(cfg.get("lengthscale", 1.0), dtype=float)
        if self.lengthscale.ndim > 1 or self.lengthscale.size == 0 or not np.isfinite(self.lengthscale).all() or (self.lengthscale <= 0).any():
            raise ValueError("lengthscale must be a positive scalar or vector")
        self.amplitude = positive(cfg.get("amplitude", 1.0), "amplitude")
        self.nugget = positive(cfg.get("nugget", 1e-8), "nugget")
        self.restarts = integer(cfg.get("optimizer_restarts", 0), "optimizer_restarts", 0)
        self.optimize = cfg.get("optimize", True)
        if not isinstance(self.optimize, bool):
            raise ValueError("optimize must be boolean")

    def validate_problem(self, problem):
        if self.lengthscale.ndim == 1 and len(self.lengthscale) != problem.d:
            raise ValueError("GP lengthscale vector must match problem dimension")

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        if self.lengthscale.ndim == 1 and len(self.lengthscale) != self.dimension:
            raise ValueError("GP lengthscale vector must match input dimension")
        z, t = self._scale_inputs(x), self._scale_targets(y)
        bounds = (1e-3, 1e3) if self.optimize else "fixed"
        kernel = ConstantKernel(self.amplitude, bounds) * RBF(self.lengthscale.tolist(), bounds)
        self.estimator = GaussianProcessRegressor(
            kernel=kernel, alpha=self.nugget, normalize_y=False,
            optimizer="fmin_l_bfgs_b" if self.optimize else None,
            n_restarts_optimizer=self.restarts if self.optimize else 0,
            random_state=self.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.estimator.fit(z, t)
        for warning in caught:
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        self._finish([self.estimator.kernel_.theta, self.estimator.alpha_],
                     kernel=str(self.estimator.kernel_),
                     warnings=[str(w.message) for w in caught],
                     log_marginal_likelihood=float(self.estimator.log_marginal_likelihood_value_),
                     variance_kind="latent_posterior_without_observation_noise",
                     nugget_units="standardized_y_squared",
                     trend="empirical_mean", input_scaling="training_mean_std")

    def predict(self, x, gradient=False):
        z = (self._input(x) - self.x_center) / self.x_scale
        kernel = self.estimator.kernel_
        length = np.asarray(kernel.k2.length_scale)
        delta = z - self.estimator.X_train_
        k = kernel.k1.constant_value * np.exp(-.5 * ((delta / length)**2).sum(axis=1))
        solved = cho_solve((self.estimator.L_, True), k, check_finite=False)
        mu = self.y_center + self.y_scale * (k @ self.estimator.alpha_)
        raw_var = self.y_scale**2 * (kernel.k1.constant_value - k @ solved)
        variance = max(float(raw_var), 1e-12)
        dm = dv = None
        if gradient:
            dk = -k[:, None] * delta / length**2 / self.x_scale
            dm = tuple(self.y_scale * (dk.T @ self.estimator.alpha_))
            dv = tuple(-2 * self.y_scale**2 * (dk.T @ solved)) if raw_var > 1e-12 else (0.0,) * self.dimension
        return Prediction(mean=float(mu), epistemic_variance=variance,
                          mean_gradient=dm, epistemic_variance_gradient=dv)
