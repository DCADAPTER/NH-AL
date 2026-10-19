"""Acquisition gradients and Reference/NH acceptance rules."""
import unittest
import math
from adaptive import EFFAcquisition, UAcquisition, accept_refinement
from metamodels import create_metamodel
from metamodels.contracts import Prediction


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
