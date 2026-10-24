"""Load a saved adaptive-learning iteration and evaluate its configured acquisition."""
import csv
import importlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metamodels import create_metamodel


@dataclass
class IterationReplay:
    run_data_dir: Path
    iteration: int
    run_config: dict
    evaluation: dict
    problem: object
    model: object
    acquisition: object
    X_train: np.ndarray
    y_train: np.ndarray


def _create_component(spec, *args):
    factory_path = spec.get("factory")
    if not isinstance(factory_path, str):
        raise ValueError("Component configuration requires a 'factory' module:class")
    try:
        module_name, class_name = factory_path.split(":")
    except ValueError as exc:
        raise ValueError(f"Invalid component factory {factory_path!r}; expected module:class") from exc
    component_class = getattr(importlib.import_module(module_name), class_name)
    return component_class(*args)


def _read_json_lines(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _verify_training_log(saved, reproduced, iteration):
    for name in ("theta", "beta", "process_variance", "nugget"):
        saved_value = saved.get(name)
        reproduced_value = reproduced.get(name)
        if saved_value is None or reproduced_value is None:
            continue
        if not np.allclose(saved_value, reproduced_value, rtol=1e-7, atol=1e-9):
            raise RuntimeError(
                f"Reproduced model {name}={reproduced_value} does not match saved "
                f"value {saved_value} at iteration {iteration}"
            )


def load_iteration_replay(run_data_dir, iteration):
    """Reconstruct the configured problem, metamodel, and acquisition snapshot."""
    run_data_dir = Path(run_data_dir).expanduser().resolve()
    if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    iteration = int(iteration)
    required = ("effective_config.json", "evaluation.jsonl", "training.csv")
    missing = [name for name in required if not (run_data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run-data files in {run_data_dir}: {missing}")

    with (run_data_dir / "effective_config.json").open(encoding="utf-8") as file:
        run_config = json.load(file)
    records = _read_json_lines(run_data_dir / "evaluation.jsonl")
    selected = [row for row in records if row.get("iteration") == iteration]
    if len(selected) != 1:
        available = sorted(row["iteration"] for row in records if "iteration" in row)
        if not available:
            raise ValueError("evaluation.jsonl contains no iteration records")
        raise ValueError(
            f"Iteration {iteration} must occur once in evaluation.jsonl; "
            f"available range is {available[0]}..{available[-1]}"
        )
    evaluation = selected[0]
    training_log = evaluation.get("training_log")
    if not isinstance(training_log, dict):
        raise ValueError(f"Iteration {iteration} has no saved training metadata")

    problem_config = run_config.get("problem")
    metamodel_config = run_config.get("metamodel")
    acquisition_config = run_config.get("acquisition")
    if not all(isinstance(cfg, dict) for cfg in
               (problem_config, metamodel_config, acquisition_config)):
        raise ValueError("effective_config.json must define problem, metamodel, and acquisition mappings")

    problem = _create_component(problem_config, problem_config)
    dimension = int(problem.d)
    n_train = int(training_log.get("n_samples", 0))
    if n_train < 2:
        raise ValueError(f"Iteration {iteration} has invalid training sample count {n_train}")
    if evaluation.get("n_true_calls") is not None and \
            n_train != int(evaluation["n_true_calls"]):
        raise ValueError("Saved iteration training count and n_true_calls do not agree")

    with (run_data_dir / "training.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = [f"x{index}" for index in range(dimension)] + ["g"]
        if not reader.fieldnames or any(name not in reader.fieldnames for name in required_columns):
            raise ValueError(f"training.csv must contain {required_columns}")
        rows = list(reader)
    if len(rows) < n_train:
        raise ValueError(
            f"Iteration requires {n_train} training rows, but training.csv has {len(rows)}"
        )

    X_train = np.asarray(
        [[float(row[f"x{index}"]) for index in range(dimension)]
         for row in rows[:n_train]],
        dtype=float,
    )
    y_train = np.asarray([float(row["g"]) for row in rows[:n_train]], dtype=float)

    replay_model_config = dict(metamodel_config)
    for name in ("theta", "nugget"):
        if name in training_log and name in replay_model_config:
            replay_model_config[name] = training_log[name]
    if "theta" in training_log and "optimize" in replay_model_config:
        replay_model_config["optimize"] = False
    model = create_metamodel(replay_model_config)
    model.fit(X_train, y_train)
    _verify_training_log(training_log, getattr(model, "training_log", {}), iteration)

    acquisition = _create_component(acquisition_config, model)
    direction = getattr(acquisition, "direction", None)
    if direction not in ("min", "max"):
        raise ValueError(
            f"Configured acquisition {type(acquisition).__name__} must declare "
            "direction='min' or 'max'"
        )
    return IterationReplay(
        run_data_dir=run_data_dir,
        iteration=iteration,
        run_config=run_config,
        evaluation=evaluation,
        problem=problem,
        model=model,
        acquisition=acquisition,
        X_train=X_train,
        y_train=y_train,
    )


def evaluate_acquisition_mc(problem, acquisition, sample_count, repetitions,
                            random_seed, batch_size, progress_callback=None):
    """Return each repetition's best configured acquisition score and candidate."""
    for name, value in (("sample_count", sample_count), ("repetitions", repetitions),
                        ("batch_size", batch_size)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    direction = getattr(acquisition, "direction", None)
    if direction not in ("min", "max"):
        raise ValueError("Acquisition must declare direction='min' or 'max'")
    score_many = getattr(acquisition, "score_many", None)
    score = getattr(acquisition, "score", None)
    if not callable(score_many) and not callable(score):
        raise TypeError("Acquisition must provide score_many(points) or score(point)")

    rng = random.Random(random_seed)
    dimension = int(problem.d)
    results = []
    for repetition in range(1, int(repetitions) + 1):
        best_value = np.inf if direction == "min" else -np.inf
        best_point = None
        remaining = int(sample_count)
        while remaining:
            count = min(int(batch_size), remaining)
            points = np.asarray(problem.sample(rng, count), dtype=float)
            if points.shape != (count, dimension) or not np.isfinite(points).all():
                raise ValueError("Problem sampler must return finite points with shape (n, d)")
            scores = np.asarray(
                score_many(points) if callable(score_many)
                else [score(point) for point in points],
                dtype=float,
            )
            if scores.shape != (count,) or not np.isfinite(scores).all():
                raise ValueError("Acquisition scores must be a finite vector matching the batch")
            index = int(np.argmin(scores) if direction == "min" else np.argmax(scores))
            value = float(scores[index])
            if (best_point is None or
                    (direction == "min" and value < best_value) or
                    (direction == "max" and value > best_value)):
                best_value = value
                best_point = points[index].copy()
            remaining -= count
        result = {
            "repetition": repetition,
            "acquisition_value": float(best_value),
            "x": best_point.tolist(),
        }
        results.append(result)
        if callable(progress_callback):
            progress_callback(result)
    return results


def predict_moments(model, points):
    """Get predictive mean and epistemic variance for a batch of points."""
    predict_many = getattr(model, "predict_moments", None)
    if callable(predict_many):
        mean, variance = predict_many(points)
        return np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    predictions = [model.predict(point) for point in points]
    return (
        np.asarray([item.require("mean") for item in predictions], dtype=float),
        np.asarray([item.require("epistemic_variance") for item in predictions], dtype=float),
    )


def score_acquisition_many(acquisition, points):
    """Evaluate a configured acquisition over a point batch."""
    score_many = getattr(acquisition, "score_many", None)
    if callable(score_many):
        values = np.asarray(score_many(points), dtype=float)
    else:
        values = np.asarray([acquisition.score(point) for point in points], dtype=float)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("Acquisition must return one finite score per point")
    return values
