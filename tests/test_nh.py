"""Hamiltonian branching into bounded local search, budgets and preflight."""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive import accept_refinement
from nh import NestedHamiltonianPrototype
from pipeline import ROOT, execute, read_config


class BoxProblem:
    bound = 5.
    def admissible(self, x):
        return all(math.isfinite(v) and abs(v) <= self.bound for v in x)
    def g(self, x):
        raise AssertionError('NH must not evaluate true g')


class Quadratic:
    direction = 'min'
    def __init__(self, target=0):
        self.target = target
        self.energy_points = self.gradient_points = 0
        self.visited = []
    def energy(self, x):
        self.energy_points += 1
        self.visited.append(list(x))
        return (x[0]-self.target)**2
    def energy_gradient(self, x):
        self.gradient_points += 1
        return [2*(x[0]-self.target)]
    def score(self, x):
        return abs(x[0]-self.target)


class ChildTests(unittest.TestCase):
    def test_real_parent_trough_spawns_terminal_lbfgsb_child(self):
        cfg = {'dt': .1, 'steps': 40, 'max_paths': 2, 'max_depth': 1,
               'energy_tolerance': .1, 'child': {'method': 'lbfgsb', 'radius': .5}}
        problem = BoxProblem()
        acquisition = Quadratic()
        proposals, diagnostics = NestedHamiltonianPrototype(cfg).refine(
            [1.], acquisition, problem, random.Random(7))
        self.assertEqual(diagnostics['hamiltonian_paths'], 1)
        self.assertEqual(diagnostics['local_search_paths'], 1)
        self.assertEqual(diagnostics['branches'], 1)
        child = diagnostics['children'][0]
        self.assertLess(child['best_energy'], child['initial_energy'])
        self.assertLess(abs(child['best_x'][0]), 1e-6)
        self.assertLessEqual(child['nfev'], 80)
        self.assertEqual(child['nfev'], child['njev'])
        point, score = accept_refinement([1.],1.,proposals,acquisition,problem.admissible)
        self.assertLess(score,1e-6)
        root, root_log = NestedHamiltonianPrototype(dict(cfg,max_paths=1)).refine(
            [1.], Quadratic(), problem, random.Random(7))
        self.assertEqual(proposals[:len(root)],root)
        self.assertEqual(diagnostics['accepted_steps'],root_log['accepted_steps'])
        self.assertTrue(all(child['bounds'][0][0] <= p[0] <= child['bounds'][0][1]
                            for p in proposals[len(root):]))
        json.dumps(diagnostics, allow_nan=False)

    def test_bound_intersection_and_strict_evaluation_budget(self):
        problem = BoxProblem(); problem.bound=1
        for budget in (1,20):
            with self.subTest(budget=budget):
                acquisition=Quadratic(target=2)
                nh=NestedHamiltonianPrototype({'child':{'method':'lbfgsb','radius':.25,'maxfun':budget}})
                proposals, log=nh._lbfgsb_child([.8],acquisition,problem)
                self.assertEqual(log['bounds'], [[.55,1.]])
                self.assertTrue(all(.55 <= p[0] <= 1 for p in acquisition.visited))
                self.assertEqual(log['nfev'],acquisition.energy_points)
                self.assertEqual(log['njev'],acquisition.gradient_points)
                self.assertLessEqual(log['nfev'],budget)
                self.assertTrue(proposals)
                if budget==1:
                    self.assertFalse(log['success'])
                    self.assertEqual(log['status'],'evaluation_limit')
                    self.assertEqual(log['best_x'],[.8])
                else:
                    self.assertAlmostEqual(log['best_x'][0],1.)

    def test_bad_gradient_does_not_emit_nonfinite_candidates(self):
        acquisition=Quadratic()
        acquisition.energy_gradient=lambda x: [float('nan')]
        nh=NestedHamiltonianPrototype({'child':{'method':'lbfgsb'}})
        proposals, log=nh._lbfgsb_child([.8],acquisition,BoxProblem())
        self.assertEqual(proposals,[])
        self.assertEqual(log['status'],'invalid_evaluation')
        self.assertFalse(log['success'])
        json.dumps(log,allow_nan=False)

    def test_hamiltonian_children_remain_selectable_and_depth_zero_disables_children(self):
        cfg={'dt':.1,'steps':40,'max_paths':2,'max_depth':1,'energy_tolerance':.1,
             'child':{'method':'hamiltonian'}}
        _, log=NestedHamiltonianPrototype(cfg).refine([1.],Quadratic(),BoxProblem(),random.Random(7))
        self.assertEqual(log['hamiltonian_paths'],2)
        self.assertEqual(log['local_search_paths'],0)
        cfg.update(max_depth=0,child={'method':'lbfgsb'})
        _, log=NestedHamiltonianPrototype(cfg).refine([1.],Quadratic(),BoxProblem(),random.Random(7))
        self.assertEqual(log['paths'],1)
        self.assertEqual(log['branches'],0)

    def test_invalid_child_configuration_fails_before_true_calls_or_outputs(self):
        cfg=read_config(ROOT/'configs/AK-MCS-U.yaml')
        cfg['evaluation']['integration_size']=16
        cfg['initial_design'].update(restarts=1,iterations=1)
        for settings in ({'method':'adam'}, {'method':'lbfgsb','radius':0},
                         {'method':'lbfgsb','maxfun':0}, {'method':'lbfgsb','gtol':float('nan')}):
            cfg['nh']['child']=settings
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'run'
                with patch('g.Problem.g') as oracle:
                    with self.assertRaises(ValueError):
                        execute(cfg,'paired',output)
                    oracle.assert_not_called()
                self.assertFalse(output.exists())
