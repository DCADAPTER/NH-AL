"""End-to-end stochastic observations and budget-limited AL-SPCE batches."""
import json
import tempfile
import unittest
from pathlib import Path

from g import Evaluator, Problem
from pipeline import ROOT, prepare, read_config


class ALSPCEPipelineTests(unittest.TestCase):
    def test_stochastic_replay_advances_response_generator(self):
        problem_cfg = {
            "type": "stochastic_linear_uniform",
            "dimension": 1,
            "beta": 1.0,
            "latent_half_width": 0.6,
            "response_seed": 42,
        }
        points = [[0.0], [0.5], [-0.5]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted = Evaluator(
                Problem(problem_cfg), root / "uninterrupted.jsonl")
            expected = [
                uninterrupted.evaluate(point, 0, "initial")
                for point in points
            ]

            replay_rows = [
                json.loads(line) for line in
                (root / "uninterrupted.jsonl").read_text(
                    encoding="utf-8").splitlines()[:2]
            ]
            resumed = Evaluator(
                Problem(problem_cfg), root / "resumed.jsonl",
                replay=replay_rows)
            for point in points[:2]:
                resumed.evaluate(point, 0, "initial")
            self.assertEqual(
                resumed.evaluate(points[2], 0, "initial"), expected[2])

    def test_stochastic_batches_respect_budget_and_write_valid_records(self):
        cfg = read_config(ROOT / "configs" / "AL-SPCE.yaml")
        cfg["problem"]["response_seed"] = 42
        cfg["initial_design"].update(size=12, seed=43)
        cfg["metamodel"].update(
            degree=1,
            max_terms=20,
            likelihood_quadrature_points=12,
            failure_quadrature_points=32,
            coefficient_draws=24,
            optimizer_maxiter=200,
            noise_candidates=3,
        )
        cfg["sampling"].update(batch_size=3, pool_size=32, max_iterations=10)
        cfg["evaluation"].update(
            integration_size=50, audit_size=1, max_true_calls=17)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "al_spce"
            summary = prepare(cfg, "reference").run(output)

            calls = [json.loads(line) for line in
                     (output / "g_calls.jsonl").read_text(encoding="utf-8").splitlines()]
            selections = [json.loads(line) for line in
                          (output / "selection.jsonl").read_text(encoding="utf-8").splitlines()]
            evaluations = [json.loads(line) for line in
                           (output / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(calls), 17)
        self.assertEqual(summary["n_true_calls"], 17)
        self.assertEqual(summary["final"]["stop_reason"], "budget_exhausted")
        self.assertEqual([len(row["x_selected_batch"]) for row in selections], [3, 2])
        self.assertEqual(evaluations[-1]["n_true_calls"], 17)
        self.assertTrue(all(row["status"] == "ok" for row in calls))

        problem = cfg["problem"]
        residuals = [
            row["y"] - problem["beta"] +
            sum(row["x"]) / (len(row["x"]) ** .5)
            for row in calls
        ]
        self.assertTrue(all(
            abs(value) <= problem["latent_half_width"]
            for value in residuals
        ))
        self.assertGreater(len(set(round(value, 8) for value in residuals)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
