"""Support vector classification of raw g: failure=-1, safe=+1."""
import numpy as np
from sklearn.svm import SVC
from .base import ArrayModel, positive
from .contracts import Prediction


class SVMClassifier(ArrayModel):
    capabilities = frozenset({"decision_function", "input_gradient", "decision_gradient"})

    def __init__(self, cfg):
        super().__init__(cfg)
        self.C = positive(cfg.get("C", 10.0), "C")
        self.kernel = cfg.get("kernel", "rbf")
        if self.kernel not in ("rbf", "linear"):
            raise ValueError("SVM supports kernel='rbf' or 'linear'")
        self.gamma = cfg.get("gamma", "scale")
        if self.gamma not in ("scale", "auto"):
            self.gamma = positive(self.gamma, "gamma")
        self.class_weight = cfg.get("class_weight", "balanced")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be null or 'balanced'")

    def fit(self, X, Y):
        x, y = self._training_data(X, Y)
        labels = np.where(y <= 0, -1, 1)
        if np.unique(labels).size != 2:
            raise ValueError("SVM requires both failure (g<=0) and safe (g>0) observations; enlarge or revise the initial design")
        z = self._scale_inputs(x)
        if self.gamma == "scale":
            self.gamma_ = 1.0 / (self.dimension * z.var()) if z.var() > 0 else 1.0
        elif self.gamma == "auto":
            self.gamma_ = 1.0 / self.dimension
        else:
            self.gamma_ = self.gamma
        self.estimator = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma_,
                             class_weight=self.class_weight, random_state=self.seed)
        self.estimator.fit(z, labels)
        self._finish([self.estimator.support_vectors_, self.estimator.dual_coef_,
                      self.estimator.intercept_, self.gamma_],
                     n_support=int(self.estimator.n_support_.sum()),
                     n_failure=int((labels == -1).sum()), n_safe=int((labels == 1).sum()),
                     label_rule="g<=0 -> -1; g>0 -> +1",
                     input_scaling="training_mean_std")

    def predict(self, x, gradient=False):
        z = (self._input(x) - self.x_center) / self.x_scale
        sv = self.estimator.support_vectors_
        weights = self.estimator.dual_coef_[0]
        if self.kernel == "linear":
            kernel = sv @ z
            derivative = (weights @ sv) / self.x_scale if gradient else None
        else:
            delta = z - sv
            kernel = np.exp(-self.gamma_ * (delta**2).sum(axis=1))
            derivative = ((weights * kernel) @ (-2 * self.gamma_ * delta)) / self.x_scale if gradient else None
        score = float(weights @ kernel + self.estimator.intercept_[0])
        return Prediction(decision_function=score,
                          decision_gradient=tuple(derivative) if gradient else None)
