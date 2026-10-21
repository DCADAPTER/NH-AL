"""Resume must preserve observations, paired seeds and deterministic selection."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import observations, repeat, repetition_config
from g import Problem
from pipeline import ROOT, read_config


class ExperimentTests(unittest.TestCase):
    def config(self):
        cfg = read_config(ROOT/'configs/AK-MCS-U.yaml')
        cfg['repetitions'] = 2
        cfg['initial_design'].update(size=4, restarts=1, iterations=5)
        cfg['sampling']['pool_size'] = 16
        cfg['evaluation'].update(integration_size=20, max_true_calls=7)
        cfg['evaluation']['stop']['min_u'] = 1e12
        cfg['metamodel']['optimizer_restarts'] = 0
        cfg['nh'].update(steps=2, max_paths=2)
        return cfg

    def test_interrupted_resume_matches_clean_and_skips_completed(self):
        cfg = self.config()
        original = Problem.g
        attempts = []
        def interrupted(problem, x):
            attempts.append(list(x))
            if len(attempts) == 13:
                raise RuntimeError('simulated interruption')
            return original(problem, x)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            out = root/'resumed'
            with patch.object(Problem, 'g', interrupted), self.assertRaisesRegex(RuntimeError, 'simulated'):
                repeat(cfg, 'paired', out)
            completed = out/'rep_001/reference/summary.json'
            saved_summary = completed.read_bytes()
            with patch.object(Problem, 'g', autospec=True, side_effect=original) as oracle:
                results = repeat(cfg, 'paired', out, resume=True)
                self.assertEqual(oracle.call_count, 28-12)
            self.assertEqual(completed.read_bytes(), saved_summary)
            self.assertEqual(results[1]['summary']['replayed_true_calls'], 5)
            clean = root/'clean'
            repeat(cfg, 'paired', clean)
            for rep in ('rep_001', 'rep_002'):
                for mode in ('reference', 'nh'):
                    resumed = observations(out/rep/mode/'g_calls.jsonl')
                    expected = observations(clean/rep/mode/'g_calls.jsonl')
                    self.assertEqual([(r['x'], r['y']) for r in resumed], [(r['x'], r['y']) for r in expected])
                self.assertEqual((out/rep/'reference/initial_design.json').read_bytes(),
                                 (out/rep/'nh/initial_design.json').read_bytes())
                self.assertEqual((out/rep/'reference/fixed_samples.json').read_bytes(),
                                 (out/rep/'nh/fixed_samples.json').read_bytes())
            self.assertNotEqual((out/'rep_001/reference/initial_design.json').read_bytes(),
                                (out/'rep_002/reference/initial_design.json').read_bytes())
            with patch.object(Problem, 'g', side_effect=AssertionError('Repeated g call')):
                repeat(cfg, 'paired', out, resume=True)
            cfg['repetitions'] = 3
            with patch.object(Problem, 'g', autospec=True, side_effect=original) as oracle:
                self.assertEqual(len(repeat(cfg, 'paired', out, resume=True)), 6)
                self.assertEqual(oracle.call_count, 14)
            cfg['nh']['dt'] *= 2
            with patch.object(Problem, 'g') as oracle, self.assertRaisesRegex(ValueError, 'configuration differs'):
                repeat(cfg, 'paired', out, resume=True)
            oracle.assert_not_called()

    def test_resume_interrupted_again_during_replay_preserves_longest_prefix(self):
        cfg = self.config()
        cfg['repetitions'] = 1
        from g import Evaluator
        original = Evaluator.evaluate
        def interrupt_after_five(oracle, x, iteration, role):
            if oracle.n_calls == 5:
                raise RuntimeError('interrupt')
            return original(oracle, x, iteration, role)
        def interrupt_after_two(oracle, x, iteration, role):
            if oracle.n_calls == 2:
                raise RuntimeError('interrupt replay')
            return original(oracle, x, iteration, role)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'repeat'
            with patch.object(Evaluator, 'evaluate', interrupt_after_five), self.assertRaises(RuntimeError):
                repeat(cfg, 'reference', out)
            with patch.object(Evaluator, 'evaluate', interrupt_after_two), self.assertRaises(RuntimeError):
                repeat(cfg, 'reference', out, resume=True)
            with patch.object(Problem, 'g', autospec=True, side_effect=Problem.g) as oracle:
                result = repeat(cfg, 'reference', out, resume=True)
                self.assertEqual(oracle.call_count, 2)
                self.assertEqual(result[0]['summary']['replayed_true_calls'], 5)

    def test_validation_and_partial_log(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                repetition_config(dict(self.config(), repetitions=invalid), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calls.jsonl'
            valid = {'status': 'ok', 'y': 2., 'cumulative_calls': 1}
            path.write_text(json.dumps(valid)+'\n{"partial":', encoding='utf-8')
            self.assertEqual(observations(path), [valid])
            path.write_text('{bad}\n'+json.dumps(valid), encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError): observations(path)

    def test_stochastic_response_seed_advances_per_repetition(self):
        cfg = self.config()
        cfg["problem"].update(type="stochastic_linear_uniform", response_seed=91)
        first = repetition_config(cfg, 0)
        second = repetition_config(cfg, 1)
        self.assertEqual(first["problem"]["response_seed"], 91)
        self.assertEqual(second["problem"]["response_seed"], 100091)
