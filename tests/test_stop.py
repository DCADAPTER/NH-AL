"""Stopping decisions and routing through the actual shared-population loop."""
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import PluginMCEvaluation
from metamodels import CompatibilityError
from metamodels.contracts import Prediction
from pipeline import ROOT, prepare, read_config, run

Stop = importlib.import_module('stop.AK-MCS-U').Stop


class StopTests(unittest.TestCase):
    def config(self):
        cfg = read_config(ROOT/'configs/AK-MCS-U.yaml')
        cfg['initial_design'].update(size=4, method='random_from_mc_population')
        cfg['metamodel']['optimize'] = False
        cfg['evaluation'].update(integration_size=8, max_true_calls=6,
                                 population_increment=8, max_population_size=16)
        return cfg

    def test_learning_precision_zero_failures_and_threshold(self):
        class Acquisition:
            def score(self, x): return x[0]
        stop = Stop({'min_u': 2, 'max_mc_cov': .05})
        for u, cov, converged, expand in [
                (1.999, .01, False, False), (2, .05, True, False),
                (2, .051, False, True), (2, None, False, True)]:
            with self.subTest(u=u, cov=cov):
                result = stop.evaluate(None, Acquisition(), [[u]], {'mc_cov': cov})
                self.assertEqual(result['converged'], converged)
                self.assertEqual(result['expand_population'], expand)
        with self.assertRaises(ValueError):
            stop.evaluate(None, Acquisition(), [[float('nan')]], {'mc_cov': .01})

    def test_observed_population_points_do_not_hold_down_min_u(self):
        class Model:
            X = [[-1]]
        class Acquisition:
            def score_many(self, points):
                return [x[0] for x in points]
        stop = Stop({'min_u': 2, 'max_mc_cov': .05})
        result = stop.evaluate(Model(), Acquisition(), [[-1], [3]], {'mc_cov': .01})
        self.assertEqual(result['population_min_u'], 3)
        self.assertEqual(result['u_scored_points'], 1)
        self.assertTrue(result['converged'])
        all_observed = stop.evaluate(Model(), Acquisition(), [[-1]], {'mc_cov': .01})
        self.assertIsNone(all_observed['population_min_u'])
        self.assertTrue(all_observed['converged'])

    def test_preflight_rejects_incompatible_model_and_separate_pool(self):
        for change in ('svm', 'separate', 'bounded', 'missing', 'threshold'):
            cfg = self.config()
            if change == 'svm': cfg['metamodel'] = {'plugin': 'svm'}
            elif change == 'separate': cfg['sampling'].pop('population')
            elif change == 'bounded': cfg['problem']['search_bound'] = 5
            elif change == 'missing': cfg['evaluation']['stop']['factory'] = 'stop.missing:Stop'
            else: cfg['evaluation']['stop']['min_u'] = 0
            with self.subTest(change=change), patch('g.Problem.g') as oracle:
                with self.assertRaises((ValueError, ModuleNotFoundError)):
                    prepare(cfg, 'reference')
                oracle.assert_not_called()

    def test_expansion_rechecks_without_true_calls_or_refitting(self):
        cfg = self.config()
        class CertainSafeModel:
            capabilities = frozenset({'mean', 'epistemic_variance'})
            snapshot = 'fitted'
            training_log = {}
            fit_count = 0
            def fit(self, X, Y): self.fit_count += 1
            def predict(self, x, gradient=False):
                return Prediction(mean=3, epistemic_variance=1)
        model = CertainSafeModel()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch('pipeline.create_metamodel', return_value=model):
                out = Path(tmp)/'run'
                result = run(cfg, 'reference', out)
            self.assertEqual(result['n_true_calls'], 4)
            self.assertEqual(model.fit_count, 1)
            self.assertFalse(result['final']['converged'])
            self.assertEqual(result['final']['stop_reason'], 'mc_population_limit')
            rows = [json.loads(s) for s in (out/'evaluation.jsonl').read_text().splitlines()]
            self.assertEqual([r['population_size'] for r in rows], [8, 16])
            initial = json.loads((out/'fixed_samples.json').read_text())['integration']
            enlarged = json.loads((out/'mc_population.json').read_text())
            self.assertEqual(enlarged[:8], initial)
            self.assertEqual(len(enlarged), 16)

    def test_actual_selection_uses_mc_population_and_budget_is_not_convergence(self):
        cfg = self.config()
        cfg['evaluation']['stop']['min_u'] = 1e12
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'run'
            result = run(cfg, 'reference', out)
            self.assertEqual(result['n_true_calls'], 6)
            self.assertFalse(result['final']['converged'])
            self.assertEqual(result['final']['stop_reason'], 'budget_exhausted')
            population = json.loads((out/'fixed_samples.json').read_text())['integration']
            choices = [json.loads(s) for s in (out/'selection.jsonl').read_text().splitlines()]
            self.assertTrue(choices)
            self.assertTrue(all(c['x_reference'] in population for c in choices))
            calls = [json.loads(s) for s in (out/'g_calls.jsonl').read_text().splitlines()]
            self.assertTrue(all(c['x'] in population for c in calls))

    def test_evaluator_reports_convergence_and_caps_separately(self):
        from adaptive import UAcquisition
        class Model:
            capabilities = frozenset({'mean', 'epistemic_variance'})
            def predict(self, x):
                return Prediction(mean=x[0], epistemic_variance=1)
        model = Model()
        cfg = self.config()['evaluation']
        cfg['stop']['max_mc_cov'] = 1
        evaluator = PluginMCEvaluation(cfg, [[-3], [3]], [])
        result = evaluator.evaluate(model, UAcquisition(model), 6)
        self.assertTrue(result['converged'])
        self.assertEqual(result['stop_reason'], 'converged_ak_mcs_u')
        evaluator.integration = [[3], [3]]
        result = evaluator.evaluate(model, UAcquisition(model), 6)
        self.assertFalse(result['converged'])
        self.assertFalse(result['expand_population'])
        self.assertEqual(result['stop_reason'], 'budget_exhausted')
