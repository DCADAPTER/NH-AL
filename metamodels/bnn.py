"""One-hidden-layer tanh BNN, diagonal Gaussian variational weight posterior.

Gaussian likelihood and Gaussian prior; reparameterized Monte Carlo likelihood
plus analytic KL(q||p), optimized by Adam. Posterior weight draws are frozen
after each fit. This is a small NumPy implementation, not a paper reproduction.
"""
import math
import numpy as np
from .base import ArrayModel, positive, integer
from .contracts import Prediction


class BayesianNeuralNetwork(ArrayModel):
    capabilities = frozenset({"mean", "epistemic_variance", "observation_noise_variance",
                              "posterior_responses", "input_gradient",
                              "mean_gradient", "epistemic_variance_gradient"})

    def __init__(self, cfg):
        super().__init__(cfg)
        self.hidden = integer(cfg.get("hidden_units", 16), "hidden_units")
        self.epochs = integer(cfg.get("epochs", 800), "epochs")
        self.train_samples = integer(cfg.get("train_samples", 4), "train_samples")
        self.posterior_samples = integer(cfg.get("posterior_samples", 64), "posterior_samples", 2)
        self.learning_rate = positive(cfg.get("learning_rate", .01), "learning_rate")
        self.prior_std = positive(cfg.get("prior_std", 1.0), "prior_std")
        self.noise_std = positive(cfg.get("noise_std", .1), "noise_std")
        self.initial_std = positive(cfg.get("initial_weight_std", .03), "initial_weight_std")
        if not math.exp(-12) <= self.initial_std <= math.exp(2):
            raise ValueError("initial_weight_std must be between exp(-12) and exp(2)")

    def _unpack(self, weights):
        cut = self.dimension * self.hidden
        w = weights[:cut].reshape(self.dimension, self.hidden)
        b = weights[cut:cut + self.hidden]
        v = weights[cut + self.hidden:cut + 2*self.hidden]
        c = weights[-1]
        return w, b, v, c

    def _forward(self, weights, x):
        w, b, v, c = self._unpack(weights)
        h = np.tanh(x @ w + b)
        return h @ v + c, h

    def _nll_gradient(self, weights, x, y):
        prediction, h = self._forward(weights, x)
        _, _, v, _ = self._unpack(weights)
        residual = prediction - y
        output_grad = residual / (len(y) * self.noise_std**2)
        hidden_grad = output_grad[:, None] * v * (1 - h**2)
        grad = np.concatenate([(x.T @ hidden_grad).ravel(), hidden_grad.sum(axis=0),
                               h.T @ output_grad, [output_grad.sum()]])
        nll = .5 * np.mean(residual**2) / self.noise_std**2 + math.log(self.noise_std)
        return float(nll), grad

    def _objective_gradient(self, mean, log_std, x, y, epsilon):
        std = np.exp(log_std)
        g_mean, g_log = np.zeros_like(mean), np.zeros_like(mean)
        nll = 0.0
        for eps in epsilon:
            value, grad = self._nll_gradient(mean + std * eps, x, y)
            nll += value / len(epsilon)
            g_mean += grad / len(epsilon)
            g_log += grad * std * eps / len(epsilon)
        prior_var = self.prior_std**2
        kl = np.sum(np.log(self.prior_std) - log_std +
                    (std**2 + mean**2) / (2 * prior_var) - .5)
        g_mean += mean / (prior_var * len(y))
        g_log += (std**2 / prior_var - 1) / len(y)
        return nll + float(kl) / len(y), g_mean, g_log

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        z, t = self._scale_inputs(x), self._scale_targets(y)
        rng = np.random.default_rng(self.seed)
        size = self.dimension*self.hidden + 2*self.hidden + 1
        mean = np.zeros(size)
        cut = self.dimension*self.hidden
        mean[:cut] = rng.normal(0, 1 / math.sqrt(self.dimension), cut)
        mean[cut+self.hidden:-1] = rng.normal(0, 1 / math.sqrt(self.hidden), self.hidden)
        log_std = np.full(size, math.log(self.initial_std))
        parameters = np.stack([mean, log_std])
        moment, square = np.zeros_like(parameters), np.zeros_like(parameters)
        for step in range(1, self.epochs + 1):
            epsilon = rng.normal(size=(self.train_samples, size))
            loss, gm, gs = self._objective_gradient(*parameters, z, t, epsilon)
            grad = np.stack([gm, gs])
            if not np.isfinite(loss) or not np.isfinite(grad).all():
                raise FloatingPointError("BNN variational training diverged")
            norm = np.linalg.norm(grad)
            grad *= min(1.0, 10.0 / max(norm, 1e-12))
            moment = .9 * moment + .1 * grad
            square = .999 * square + .001 * grad**2
            parameters -= self.learning_rate * (moment / (1 - .9**step)) / (
                np.sqrt(square / (1 - .999**step)) + 1e-8)
            parameters[1] = np.clip(parameters[1], -12, 2)
        self.weight_mean, self.weight_log_std = parameters
        prediction_rng = np.random.default_rng(self.seed + 1)
        self.weight_draws = self.weight_mean + np.exp(self.weight_log_std) * prediction_rng.normal(
            size=(self.posterior_samples, size))
        final_epsilon = (self.weight_draws - self.weight_mean) / np.exp(self.weight_log_std)
        final_loss, _, _ = self._objective_gradient(*parameters, z, t, final_epsilon)
        self._finish([parameters, self.weight_draws], epochs=self.epochs,
                     negative_elbo_per_observation=float(final_loss),
                     objective_constant="Gaussian log(sqrt(2*pi)) omitted",
                     inference="mean_field_gaussian_variational",
                     noise_std_standardized_y=self.noise_std,
                     posterior_samples=self.posterior_samples,
                     posterior_draws_frozen=True)

    def predict(self, x, gradient=False):
        z = (self._input(x) - self.x_center) / self.x_scale
        cut = self.dimension*self.hidden
        w = self.weight_draws[:, :cut].reshape(-1, self.dimension, self.hidden)
        b = self.weight_draws[:, cut:cut+self.hidden]
        v = self.weight_draws[:, cut+self.hidden:-1]
        h = np.tanh(np.einsum("d,sdh->sh", z, w) + b)
        responses = self.y_center + self.y_scale * ((h * v).sum(axis=1) + self.weight_draws[:, -1])
        mu = float(responses.mean())
        raw_var = float(responses.var(ddof=1))
        dm = dv = None
        if gradient:
            derivatives = self.y_scale * np.einsum("sh,sdh->sd", v * (1-h**2), w) / self.x_scale
            dm = tuple(derivatives.mean(axis=0))
            dv = tuple(2 * ((responses-mu)[:, None] * derivatives).sum(axis=0) /
                       (self.posterior_samples-1)) if raw_var > 1e-12 else (0.0,) * self.dimension
        return Prediction(mean=mu, epistemic_variance=max(raw_var, 1e-12),
                          observation_noise_variance=(self.y_scale*self.noise_std)**2,
                          posterior_responses=tuple(responses), mean_gradient=dm,
                          epistemic_variance_gradient=dv)
