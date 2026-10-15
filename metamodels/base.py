"""Numerical validation, preprocessing and reproducible fitted-state identity."""
import hashlib
import json
import math
import numpy as np


def positive(value, name, allow_zero=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class ArrayModel:
    snapshot = None

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.snapshot = None
        self.training_log = {}
        self.seed = integer(cfg.get("seed", 0), "seed", 0)

    def _training_data(self, X, Y):
        self.snapshot = None
        self.training_log = {}
        x, y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if x.ndim != 2 or min(x.shape) < 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected X shape (n,d), Y shape (n,), with n,d >= 1")
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training requires >= 2 finite observations")
        self.dimension = x.shape[1]
        self.X, self.Y = x.copy(), y.copy()
        return x, y

    def _input(self, x):
        if self.snapshot is None:
            raise RuntimeError("Model is not fitted; call fit(X,Y) first")
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError(f"Expected a finite input of shape ({self.dimension},)")
        return x

    def _scale_inputs(self, x):
        self.x_center = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale = np.where(self.x_scale > 1e-12, self.x_scale, 1.0)
        return (x - self.x_center) / self.x_scale

    def _scale_targets(self, y):
        self.y_center = float(y.mean())
        self.y_scale = max(float(y.std()), 1e-8)
        return (y - self.y_center) / self.y_scale

    def _finish(self, state, **log):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        payload = [type(self).__name__, self.cfg, self.X, self.Y, state]
        raw = json.dumps(payload, sort_keys=True, default=encode, allow_nan=False)
        self.snapshot = hashlib.sha256(raw.encode()).hexdigest()[:20]
        self.training_log = {"n_samples": len(self.X), "dimension": self.dimension, **log}

    def predict_many(self, X, gradient=False):
        return [self.predict(x, gradient=gradient) for x in X]
