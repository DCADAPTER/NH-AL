"""Explicit registry plus module:class adapters; no import-time heavy dependencies."""
import importlib

_REGISTRY = {
    "gp": ("metamodels.gp:KrigingGP", "RBF Gaussian process regression"),
    "svm": ("metamodels.svm:SVMClassifier", "RBF/linear support vector classification"),
    "pce": ("metamodels.pce:PolynomialChaos", "Hermite PCE with optional bootstrap"),
    "spce": ("metamodels.spce:StochasticPCE", "Stochastic PCE with MLE coefficient uncertainty"),
    "bnn": ("metamodels.bnn:BayesianNeuralNetwork", "Mean-field variational Bayesian neural network"),
}


def list_metamodels():
    return {name: {"factory": entry[0], "description": entry[1]}
            for name, entry in _REGISTRY.items()}


def create_metamodel(cfg):
    if ("plugin" in cfg) == ("factory" in cfg):
        raise ValueError("metamodel requires exactly one of 'plugin' or 'factory'")
    if "plugin" in cfg:
        name = cfg["plugin"].lower()
        if name not in _REGISTRY:
            raise ValueError(f"Unknown metamodel {name!r}; choose {sorted(_REGISTRY)}")
        path = _REGISTRY[name][0]
    else:
        path = cfg["factory"]
    try:
        module, name = path.split(":")
    except (ValueError, AttributeError) as exc:
        raise ValueError("factory must be 'module:class'") from exc
    try:
        return getattr(importlib.import_module(module), name)(cfg)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Dependency {exc.name!r} missing for {path}; install requirements.txt") from exc
