"""Independent Kriging equations, gradients and paired initial-design replay."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from adaptive import UAcquisition, PoolSelector, BatchedPoolSelector
from g import Problem
from initial_design import initial_design
from metamodels.ordinary_kriging import OrdinaryKriging
from pipeline import ROOT, execute, read_config


class RastriginTests(unittest.TestCase):
    def test_problem(self):
        p = Problem({'type': 'rastrigin', 'dimension': 2, 'search_bound': None})
        self.assertEqual(p.g([0, 0]), 20)
        self.assertAlmostEqual(p.g([.5, .5]), -.5)
        self.assertAlmostEqual(p.g([1, 1]), 18)
        with self.assertRaises(ValueError):
            Problem({'type': 'rastrigin', 'dimension': 3})

    def test_ordinary_kriging_blup_and_input_derivatives(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(12, 2))
        Y = np.sin(X[:, 0]) + X[:, 1]**2
        model = OrdinaryKriging({'theta': [.7, 1.3], 'optimize': False})
        model.fit(X, Y)
        x = np.array([.3, -.4])
        z = (X-X.mean(axis=0))/X.std(axis=0, ddof=1)
        q = (x-X.mean(axis=0))/X.std(axis=0, ddof=1)
        R = np.exp(-cdist(z*np.sqrt(model.theta), z*np.sqrt(model.theta), 'sqeuclidean')) + model.nugget*np.eye(len(X))
        r = np.exp(-np.sum(model.theta*(z-q)**2, axis=1))
        # Solve the unbiased kriging saddle-point system independently.
        augmented = np.block([[R, np.ones((len(X), 1))], [np.ones((1,len(X))), np.zeros((1,1))]])
        weights = np.linalg.solve(augmented, np.r_[r, 1])[:len(X)]
        beta = np.ones(len(X)) @ np.linalg.solve(R, Y) / (np.ones(len(X)) @ np.linalg.solve(R, np.ones(len(X))))
        residual = Y-beta
        sigma2 = residual @ np.linalg.solve(R,residual)/len(X)
        variance = sigma2*(1 + weights @ R @ weights - 2*weights @ r)
        pred = model.predict(x, gradient=True)
        self.assertAlmostEqual(pred.mean, weights @ Y, places=9)
        self.assertAlmostEqual(pred.epistemic_variance, variance, places=9)
        h = 1e-5
        for j in range(2):
            e = np.eye(2)[j]*h
            plus, minus = model.predict(x+e), model.predict(x-e)
            self.assertAlmostEqual(pred.mean_gradient[j], (plus.mean-minus.mean)/(2*h), places=5)
            self.assertAlmostEqual(pred.epistemic_variance_gradient[j],
                                   (plus.epistemic_variance-minus.epistemic_variance)/(2*h), places=5)
        mu, var = model.predict_moments([x, x+.2])
        np.testing.assert_allclose(mu, [model.predict(x).mean, model.predict(x+.2).mean])
        np.testing.assert_allclose(var, [model.predict(x).epistemic_variance, model.predict(x+.2).epistemic_variance])
        np.testing.assert_allclose(model.predict_moments(X)[0], Y, atol=1e-7)

    def test_profile_likelihood_optimization_improves_objective(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(16, 2))
        Y = np.sin(X[:,0]*2) + np.cos(X[:,1])
        fixed = OrdinaryKriging({'optimize': False})
        fitted = OrdinaryKriging({'optimizer_restarts': 1})
        fixed.fit(X,Y); fitted.fit(X,Y)
        self.assertLessEqual(fitted.training_log['objective'], fixed.training_log['objective']+1e-9)
        self.assertTrue(np.isfinite(fitted.predict([0,0]).epistemic_variance))

    def test_optimized_lhd_strata_seed_and_replay(self):
        problem = Problem({'type': 'rastrigin', 'search_bound': None})
        cfg = {'size': 10, 'method': 'optimal_lhd', 'restarts': 3, 'iterations': 150}
        points, meta = initial_design(cfg, problem, 19, [])
        again, again_meta = initial_design(cfg, problem, 19, [])
        self.assertEqual(points, again)
        np.testing.assert_array_equal(np.sort(np.floor(ndtr(points)*10).astype(int),axis=0),
                                      np.repeat(np.arange(10)[:,None],2,axis=1))
        self.assertNotEqual(points, initial_design(cfg, problem, 20, [])[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'initial.json'
            path.write_text(json.dumps(dict(meta,X=points)), encoding='utf-8')
            replay, _ = initial_design(dict(cfg,input_file=str(path)), problem, 999, [])
            self.assertEqual(points,replay)
            with self.assertRaises(ValueError):
                initial_design(dict(cfg,size=11,input_file=str(path)), problem, 999, [])

    def test_batched_selection_matches_scalar(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(10,2)).tolist()
        p = Problem({'type':'rastrigin','search_bound':None})
        m = OrdinaryKriging({'optimize':False,'batch_size':7})
        m.fit(X,[p.g(x) for x in X])
        pool = X + rng.normal(size=(25,2)).tolist()
        a = PoolSelector({}).select(pool,UAcquisition(m),X,p)
        b = BatchedPoolSelector({}).select(pool,UAcquisition(m),X,p)
        self.assertEqual(a[0],b[0]); self.assertAlmostEqual(a[1],b[1])

    def test_paired_saved_initial_points_and_observations_are_identical(self):
        cfg = read_config(ROOT/'configs/AK-MCS-U.yaml')
        cfg['initial_design'].update(size=10,restarts=2,iterations=40)
        cfg['metamodel'].update(optimize=False,batch_size=17)
        cfg['evaluation'].update(integration_size=64,max_true_calls=11)
        cfg['evaluation']['stop']['min_u']=1e12
        cfg['nh'].update(steps=1,max_paths=1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'paired'
            summaries = execute(cfg,'paired',out)
            reference = json.loads((out/'reference/initial_design.json').read_text())
            nh = json.loads((out/'nh/initial_design.json').read_text())
            self.assertEqual(reference,nh)
            self.assertEqual(len(reference['X']),10)
            self.assertEqual(len(reference['Y']),10)
            self.assertTrue(all(r['n_true_calls']==11 for r in summaries.values()))
            cfg['initial_design']['input_file']=str(out/'reference/initial_design.json')
            cfg['seed']=123
            replay_out=Path(tmp)/'replay'
            execute(cfg,'reference',replay_out)
            replay=json.loads((replay_out/'initial_design.json').read_text())
            self.assertEqual(reference['X'],replay['X'])
            self.assertEqual(reference['Y'],replay['Y'])
