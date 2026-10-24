"""Tests for plugin-driven iteration replay and acquisition Monte Carlo."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive import EFFAcquisition
from g import Problem
from metamodels import create_metamodel
from replay import evaluate_acquisition_mc, load_iteration_replay


class ReplayTests(unittest.TestCase):
    def _write_run(self, directory):
        problem_config = {
            "factory": "g:Problem",
            "type": "linear_normal",
            "dimension": 1,
            "beta": 1.0,
        }
        model_config = {
            "factory": "metamodels.ordinary_kriging:OrdinaryKriging",
            "theta": [1.0],
            "theta_bounds": [0.001, 1000.0],
            "nugget": 1e-6,
            "optimize": False,
            "optimizer_restarts": 0,
        }
        acquisition_config = {
            "factory": "adaptive:EFFAcquisition",
            "type": "EFF",
            "direction": "max",
        }
        problem = Problem(problem_config)
        X = np.asarray([[-1.0], [0.0], [1.0]])
        y = np.asarray([problem.g(point.tolist()) for point in X])
        model = create_metamodel(model_config)
        model.fit(X, y)
        config = {
            "problem": problem_config,
            "metamodel": model_config,
            "acquisition": acquisition_config,
        }
        (directory / "effective_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        with (directory / "evaluation.jsonl").open("w", encoding="utf-8") as file:
            file.write(json.dumps({
                "iteration": 0,
                "n_true_calls": len(X),
                "model_snapshot": model.snapshot,
                "training_log": model.training_log,
            }) + "\n")
        with (directory / "training.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["x0", "g"])
            writer.writerows([[point[0], value] for point, value in zip(X, y)])

    def test_loader_constructs_configured_eff_plugin_and_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            self._write_run(run_dir)
            replay = load_iteration_replay(run_dir, 0)

        self.assertIsInstance(replay.acquisition, EFFAcquisition)
        self.assertEqual(replay.acquisition.direction, "max")
        self.assertEqual(replay.model.training_log["n_samples"], 3)
        self.assertEqual(replay.X_train.shape, (3, 1))

    def test_mc_uses_configured_minimize_or_maximize_direction(self):
        class FixedProblem:
            d = 1

            def __init__(self):
                self.offset = 0

            def sample(self, rng, count):
                values = [1.0, 2.0, 3.0]
                points = [
                    [values[(self.offset + index) % len(values)]]
                    for index in range(count)
                ]
                self.offset += count
                return points

        class Scores:
            def __init__(self, direction):
                self.direction = direction

            def score_many(self, points):
                return np.asarray(points)[:, 0]

        problem = FixedProblem()
        maximum = evaluate_acquisition_mc(
            problem, Scores("max"), 3, 1, random_seed=1, batch_size=2
        )
        minimum = evaluate_acquisition_mc(
            problem, Scores("min"), 3, 1, random_seed=1, batch_size=2
        )
        self.assertEqual(maximum[0]["acquisition_value"], 3.0)
        self.assertEqual(maximum[0]["x"], [3.0])
        self.assertEqual(minimum[0]["acquisition_value"], 1.0)
        self.assertEqual(minimum[0]["x"], [1.0])


if __name__ == "__main__":
    unittest.main()
