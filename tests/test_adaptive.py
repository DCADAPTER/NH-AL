"""Acquisition gradients and Reference/NH acceptance rules."""
import unittest
from adaptive import UAcquisition, accept_refinement
from metamodels import create_metamodel


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

