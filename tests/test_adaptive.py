"""Acquisition gradients and Reference/NH acceptance rules."""
import unittest
import math
import numpy as np
from adaptive import ALSPCEAcquisition, EFFAcquisition, UAcquisition, WeightedKMedoidsBatchSelector, accept_refinement
from metamodels import create_metamodel
from metamodels.contracts import Prediction
from metamodels.spce import StochasticPCE
from g import Problem


class AdaptiveTests(unittest.TestCase):
    def test_energy_gradient(self):
        model=create_metamodel({"plugin":"gp","optimize":False,"lengthscale":1.1,"nugget":1e-8})
        model.fit([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]],
                  [1.0,2.0,-.5,1.2,.3])
        acq=UAcquisition(model)
        for x in ([.2,-.3],[-.4,.25],[.6,.5]):
            grad=acq.energy_gradient(x)
            for j in range(2):
                lo=list(x);hi=list(x);h=1e-6
                lo[j]-=h;hi[j]+=h
                numerical=(acq.energy(hi)-acq.energy(lo))/(2*h)
                self.assertAlmostEqual(grad[j],numerical,delta=2e-4*max(1,abs(numerical)))

    def test_eff_expected_feasibility_and_batch_scores(self):
        class Model:
            def predict(self, x, gradient=False):
                return Prediction(mean=x[0], epistemic_variance=1.0)
            def predict_moments(self, points):
                import numpy as np
                return np.asarray([x[0] for x in points]), np.ones(len(points))

        acq = EFFAcquisition(Model())
        expected_at_limit = (2*math.erf(2/math.sqrt(2))
                             + 2*math.exp(-2)/math.sqrt(2*math.pi)
                             - 2/math.sqrt(2*math.pi))
        self.assertAlmostEqual(acq.score([0]), expected_at_limit, places=12)
        self.assertAlmostEqual(acq.score([.5]), acq.score([-.5]), places=12)
        points = [[-.5], [0], [.5]]
        import numpy as np
        np.testing.assert_allclose(acq.score_many(points),
                                   [acq.score(x) for x in points])
        self.assertLess(acq.score([10]), acq.score([0]))

    def test_eff_energy_gradient(self):
        model=create_metamodel({"plugin":"gp","optimize":False,"lengthscale":1.1,"nugget":1e-8})
        model.fit([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]],
                  [1.0,2.0,-.5,1.2,.3])
        acq=EFFAcquisition(model)
        for x in ([.2,-.3],[-.4,.25],[.6,.5]):
            grad=acq.energy_gradient(x)
            for j in range(2):
                lo=list(x);hi=list(x);h=1e-6
                lo[j]-=h;hi[j]+=h
                numerical=(acq.energy(hi)-acq.energy(lo))/(2*h)
                self.assertAlmostEqual(grad[j],numerical,delta=2e-4*max(1,abs(numerical)))

    def test_al_spce_acquisition_gradient(self):
        rng = np.random.default_rng(27)
        x = rng.normal(size=(80, 1))
        latent = rng.uniform(size=len(x))
        y = .1 + .5*x[:, 0] + .4*(2*latent-1)
        model = StochasticPCE({
            "degree": 1, "coefficient_draws": 32,
            "likelihood_quadrature_points": 24,
            "failure_quadrature_points": 80,
        })
        model.fit(x, y)
        acquisition = ALSPCEAcquisition(model)
        point = [.1]
        gradient = acquisition.energy_gradient(point)[0]
        h = 1e-5
        numerical = (acquisition.energy([point[0]+h]) -
                     acquisition.energy([point[0]-h]))/(2*h)
        self.assertAlmostEqual(gradient, numerical, delta=1e-5)
        self.assertGreater(acquisition.score([0]), 0)

    def test_weighted_kmedoids_selects_a_distinct_batch(self):
        import numpy as np
        class Scores:
            direction = "max"
            def score_many(self, points):
                return np.asarray([x[0] for x in points])

        points = [[0.0], [0.1], [4.0], [4.1], [8.0], [8.1]]
        selector = WeightedKMedoidsBatchSelector(
            {"batch_size": 3, "min_distance": .01})
        selected = selector.select_batch(
            points, Scores(), [], Problem({"type": "linear_normal", "dimension": 1}))
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({tuple(x) for x, _ in selected}), 3)
        self.assertTrue(all(point in points for point, _ in selected))


    def test_min_max_acceptance_and_constraint(self):
        class Score:
            direction="max"
            def score(self,x):return x[0]
        score=Score()
        x,value=accept_refinement([1],1,[[3],[2],[.5]],score,lambda x:x[0]<=2)
        self.assertEqual((x,value),([2],2))
        score.direction="min"
        x,value=accept_refinement([1],1,[[3],[2]],score,lambda x:True)
        self.assertEqual((x,value),([1],1))
