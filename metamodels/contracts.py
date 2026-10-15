"""Shared prediction contract and preflight checks (never fits or calls true g)."""
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """One input, in original g units; unsupported fields are None, never fabricated.

    Gradients are with respect to the input passed to predict(). SVM decision
    scores are oriented so <= 0 denotes failure. Ensemble/posterior samples are
    frozen between fit calls. observation_noise_variance is separate from
    epistemic_variance and must not be used as learning uncertainty.
    """
    mean: float | None = None
    epistemic_variance: float | None = None
    observation_noise_variance: float | None = None
    decision_function: float | None = None
    ensemble_responses: tuple[float, ...] | None = None
    ensemble_variance: float | None = None
    posterior_responses: tuple[float, ...] | None = None
    mean_gradient: tuple[float, ...] | None = None
    epistemic_variance_gradient: tuple[float, ...] | None = None
    decision_gradient: tuple[float, ...] | None = None

    def require(self, field: str):
        value = getattr(self, field)
        if value is None:
            raise CompatibilityError(f"Prediction does not provide {field!r}")
        return value


@runtime_checkable
class Metamodel(Protocol):
    capabilities: frozenset[str]
    snapshot: str | None
    training_log: Mapping

    def fit(self, X: Sequence[Sequence[float]], Y: Sequence[float]) -> None: ...
    def predict(self, x: Sequence[float], gradient: bool = False) -> Prediction: ...


class CompatibilityError(ValueError):
    """An unsupported plugin combination, detected before expensive evaluation."""


def require_capabilities(provider, required, consumer):
    if not isinstance(required, (set, frozenset)):
        raise CompatibilityError(f"{consumer}: required_capabilities must be a set")
    provided = getattr(provider, "capabilities", None)
    if not isinstance(provided, (set, frozenset)):
        raise CompatibilityError(f"{type(provider).__name__} must declare capabilities")
    missing = required - provided
    if missing:
        raise CompatibilityError(
            f"{consumer} requires {sorted(missing)}; "
            f"{type(provider).__name__} provides {sorted(provided)}")


def _requirements(consumer, name):
    required = getattr(consumer, name, None)
    if required is None:
        raise CompatibilityError(f"{type(consumer).__name__} must declare {name}")
    return required


def validate_pipeline(model, acquisition, evaluator, refiner=None, problem=None):
    """Validate declared capabilities and callable contracts without predicting."""
    for name in ("fit", "predict"):
        if not callable(getattr(model, name, None)):
            raise CompatibilityError(f"Metamodel must implement {name}()")
    for name in ("snapshot", "training_log"):
        if not hasattr(model, name):
            raise CompatibilityError(f"Metamodel must expose {name}")
    require_capabilities(model, _requirements(acquisition, "required_capabilities"),
                         type(acquisition).__name__)
    require_capabilities(model, _requirements(evaluator, "required_capabilities"),
                         type(evaluator).__name__)
    alternatives = getattr(evaluator, "required_any_capabilities", frozenset())
    if alternatives and not (alternatives & model.capabilities):
        raise CompatibilityError(f"{type(evaluator).__name__} requires one of {sorted(alternatives)}")
    if not callable(getattr(acquisition, "score", None)):
        raise CompatibilityError("Acquisition must implement score()")
    if getattr(acquisition, "direction", None) not in ("min", "max"):
        raise CompatibilityError("Acquisition direction must be min or max")
    kinds = getattr(evaluator, "supported_acquisition_kinds", None)
    if kinds is not None and getattr(acquisition, "kind", None) not in kinds:
        raise CompatibilityError(f"{type(evaluator).__name__} requires acquisition kind {sorted(kinds)}")
    if refiner is not None:
        require_capabilities(acquisition, _requirements(refiner, "required_capabilities"),
                             type(refiner).__name__)
        required = _requirements(acquisition, "gradient_required_capabilities")
        require_capabilities(model, required, type(refiner).__name__ + " via " + type(acquisition).__name__)
        for name in ("energy", "energy_gradient"):
            if not callable(getattr(acquisition, name, None)):
                raise CompatibilityError(f"NH requires callable acquisition.{name}()")
        if not callable(getattr(refiner, "refine", None)):
            raise CompatibilityError("Refiner must implement refine()")
    if problem is not None and callable(getattr(model, "validate_problem", None)):
        model.validate_problem(problem)
