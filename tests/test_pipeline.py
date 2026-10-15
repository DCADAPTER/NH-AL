"""Integration tests for config -> plugins -> selection -> true g -> evaluation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive import UAcquisition
from evaluation import MCEvaluation
from metamodels import CompatibilityError, create_metamodel, validate_pipeline
from nh import NestedHamiltonianPrototype
from pipeline import DEFAULT_CONFIG, ROOT, execute, prepare, read_config, run


class PipelineTests(unittest.TestCase):
    def test_incompatible_combinations_never_call_g(self):
        for plugin in ("svm", "pce"):
            cfg = read_config(ROOT/"configs/Kriging_GP.yaml")
            cfg["metamodel"] = {"plugin": plugin}
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/"bad"
                with patch("g.Problem.g", side_effect=AssertionError("true g called")) as oracle:
                    with self.assertRaises(CompatibilityError):
                        run(cfg, "nh", output)
                    oracle.assert_not_called()
                    self.assertFalse(output.exists())


    def test_u_evaluation_rejects_non_u_even_with_gp(self):
        cfg = read_config(ROOT/"configs/Kriging_GP.yaml")
        cfg["acquisition"] = {"factory": "adaptive:MeanBoundaryAcquisition"}
        with patch("g.Problem.g") as oracle, self.assertRaises(CompatibilityError):
            prepare(cfg, "reference")
        oracle.assert_not_called()


    def test_nh_requires_gradient_capabilities_and_callable(self):
        cfg = {"max_true_calls": 10, "max_mc_cov": .2, "min_audit_u": 2}
        class MomentOnly:
            capabilities = frozenset({"mean", "epistemic_variance"})
            snapshot = None
            training_log = {}
            def fit(self, X, Y): pass
            def predict(self, x, gradient=False): raise AssertionError("predict during preflight")
        model = MomentOnly()
        acq = UAcquisition(model)
        evaluator = MCEvaluation(cfg, [], [])
        validate_pipeline(model, acq, evaluator)
        with self.assertRaisesRegex(CompatibilityError, "gradient"):
            validate_pipeline(model, acq, evaluator, NestedHamiltonianPrototype({}))
        acq = UAcquisition(create_metamodel({"plugin": "gp"}))
        acq.energy_gradient = None
        with self.assertRaisesRegex(CompatibilityError, "callable"):
            validate_pipeline(create_metamodel({"plugin": "gp"}), acq, evaluator, NestedHamiltonianPrototype({}))


    def test_all_method_yamls_reference_and_nh_runs(self):
        for name in ("Kriging_GP", "SVM", "PCE", "BNN"):
            cfg = read_config(ROOT/f"configs/{name}.yaml")
            cfg["initial_design"]["size"] = 12
            cfg["sampling"]["pool_size"] = 16
            cfg["evaluation"].update(integration_size=24, audit_size=12, max_true_calls=14)
            if "min_audit_u" in cfg["evaluation"]:
                cfg["evaluation"]["min_audit_u"] = 1e12
            cfg["nh"].update(steps=2, max_paths=1)
            if name == "BNN":
                cfg["metamodel"].update(epochs=40, posterior_samples=12)
            if name == "Kriging_GP":
                cfg["metamodel"]["optimize"] = False
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                comparison = Path(tmp)/"comparison"
                summaries = execute(cfg, "paired", comparison)
                for mode in ("reference", "nh"):
                    output = comparison/mode
                    summary = summaries[mode]
                    self.assertEqual(summary["n_true_calls"], 14)
                    self.assertEqual(summary["final"]["stop_reason"], "budget_exhausted")
                    self.assertIn("training_log", summary["final"])
                    records = [json.loads(row) for row in (output/"selection.jsonl").read_text().splitlines()]
                    self.assertEqual(len(records), 2)
                    self.assertTrue(all(row["acquisition_gain"] >= 0 for row in records))
                def first(mode):
                    return json.loads((comparison/mode/"selection.jsonl").read_text().splitlines()[0])
                self.assertEqual(first("reference")["snapshot"], first("nh")["snapshot"])
                self.assertEqual(first("reference")["x_reference"], first("nh")["x_reference"])


    def test_paired_true_calls_and_original_score(self):
        cfg=read_config(DEFAULT_CONFIG)
        cfg["metamodel"]["optimize"]=False
        cfg["initial_design"]["size"]=6
        cfg["evaluation"].update(max_true_calls=12,integration_size=64,audit_size=32,min_audit_u=1e9)
        cfg["sampling"]["pool_size"]=32
        cfg["nh"].update(steps=5,max_paths=2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)
            self.assertTrue(out.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
            for mode in ("reference","nh"):
                summary=run(cfg,mode,out/mode)
                self.assertEqual(summary["n_true_calls"],12)
                self.assertEqual(summary["final"]["stop_reason"],"budget_exhausted")
                calls=[json.loads(s) for s in (out/mode/"g_calls.jsonl").read_text().splitlines()]
                choices=[json.loads(s) for s in (out/mode/"selection.jsonl").read_text().splitlines()]
                self.assertEqual(len(calls),12)
                self.assertEqual(len(choices),6)
                self.assertTrue(all(r["acquisition_gain"]>=0 for r in choices))
                self.assertTrue(all(r["sampling_seconds"]>=r["reference_seconds"] for r in choices))
                for call,choice in zip(calls[6:],choices):
                    self.assertEqual(call["x"],choice["x_selected"])
                self.assertTrue(all(r["status"]=="ok" for r in calls))
            a=[json.loads(s) for s in (out/"reference"/"selection.jsonl").read_text().splitlines()][0]
            b=[json.loads(s) for s in (out/"nh"/"selection.jsonl").read_text().splitlines()][0]
            self.assertEqual(a["snapshot"],b["snapshot"])
            self.assertEqual(a["x_reference"],b["x_reference"])
            self.assertEqual(a["a_reference"],b["a_reference"])
            self.assertEqual((out/"reference"/"fixed_samples.json").read_text(),
                             (out/"nh"/"fixed_samples.json").read_text())


    def test_paper_registry_is_not_claimed_runnable(self):
        papers=read_config(ROOT/"papers.yaml")["profiles"]
        self.assertEqual({p["id"] for p in papers},{f"P{i:02}" for i in range(1,31)})
        self.assertTrue(all(not p["implementation"]["paper_reproduction_verified"] for p in papers))

    def test_paired_preflight_rejects_nh_before_reference_calls_g(self):
        class MomentOnly:
            capabilities = frozenset({"mean", "epistemic_variance"})
            snapshot = None
            training_log = {}
            def fit(self, X, Y):
                raise AssertionError("fit during preflight")
            def predict(self, x, gradient=False):
                raise AssertionError("predict during preflight")

        cfg = read_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"paired"
            with patch("pipeline.create_metamodel", side_effect=lambda cfg: MomentOnly()):
                with patch("g.Problem.g") as oracle:
                    with self.assertRaisesRegex(CompatibilityError, "gradient"):
                        execute(cfg, "paired", output)
                    oracle.assert_not_called()
                    self.assertFalse(output.exists())

    def test_cli_uses_method_configs_for_both_modes(self):
        from run import main
        for name in ("Kriging_GP", "SVM", "PCE", "BNN"):
            output = io.StringIO()
            with self.subTest(name=name), contextlib.redirect_stdout(output):
                with patch("g.Problem.g") as oracle:
                    main(["--config", str(ROOT/f"configs/{name}.yaml"),
                          "--mode", "paired", "--check-config"])
                    oracle.assert_not_called()
                result = json.loads(output.getvalue())
                self.assertTrue(result["valid"])
                self.assertEqual(set(result["runs"]), {"reference", "nh"})

        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--check-config"])
        self.assertEqual(json.loads(output.getvalue())["runs"]["reference"]["model"], "KrigingGP")


    def test_acquisition_is_bound_to_each_fitted_snapshot(self):
        from pipeline import factory as original_factory

        created = []

        class SnapshotAcquisition(UAcquisition):
            def __init__(self, model):
                super().__init__(model)
                self.frozen_snapshot = model.snapshot
                created.append(self.frozen_snapshot)

            def score(self, x):
                if self.frozen_snapshot is None or self.frozen_snapshot != self.model.snapshot:
                    raise AssertionError("Acquisition must use the current fitted snapshot")
                return super().score(x)

        def make_adapter(spec, *args):
            if spec["factory"] == "adaptive:UAcquisition":
                return SnapshotAcquisition(*args)
            return original_factory(spec, *args)

        cfg = read_config(DEFAULT_CONFIG)
        cfg["metamodel"]["optimize"] = False
        cfg["initial_design"]["size"] = 6
        cfg["sampling"]["pool_size"] = 8
        cfg["evaluation"].update(integration_size=16, audit_size=8,
                                 max_true_calls=8, min_audit_u=1e12)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch("pipeline.factory", side_effect=make_adapter):
                summary = run(cfg, "reference", Path(tmp)/"run")
        self.assertEqual(summary["n_true_calls"], 8)
        self.assertIsNone(created[0])  # Unfitted preflight: no scoring is allowed.
        self.assertEqual(len(created), 4)
        self.assertTrue(all(snapshot is not None for snapshot in created[1:]))
        self.assertEqual(len(set(created[1:])), 3)


    def test_cli_paper_listing_uses_utf8_on_windows(self):
        import os
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT/"run.py"), "--list-papers"],
            env=dict(os.environ, PYTHONIOENCODING="cp949"),
            capture_output=True, encoding="utf-8", check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 30)
        self.assertTrue(lines[-1].startswith("P30 "))

