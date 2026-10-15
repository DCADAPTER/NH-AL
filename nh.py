"""Illustrative nested Hamiltonian SEARCH, not the user's verified NHSVM implementation.
No sampling-invariance claim. No true-model evaluations. Replace via nh.factory.
Children spawn at a sampled potential trough (previous down, current up).
"""
import math

class NestedHamiltonianPrototype:
    required_capabilities = frozenset({"energy", "energy_gradient"})
    def __init__(self,cfg):
        self.cfg=cfg
    def refine(self,x_ref,acquisition,problem,rng):
        c=self.cfg
        mass=float(c.get("mass",1.0))
        dt0=float(c.get("dt",.03))
        steps=int(c.get("steps",15))
        max_paths=int(c.get("max_paths",4))
        max_depth=int(c.get("max_depth",2))
        max_retries=int(c.get("max_retries",6))
        tolerance=float(c.get("energy_tolerance",1e-3))
        temperature=float(c.get("temperature",.2))
        if min(mass,dt0,tolerance,temperature)<=0 or steps<1 or max_paths<1 or max_retries<0:
            raise ValueError("Invalid NH configuration")
        queue=[(list(x_ref),0)]
        proposals=[]
        diagnostics={"paths":0,"branches":0,"rejected_steps":0,"accepted_steps":0}
        while queue and diagnostics["paths"]<max_paths:
            x,depth=queue.pop(0)
            p=[rng.gauss(0,math.sqrt(mass*temperature)) for _ in x]
            values=[acquisition.energy(x)]
            diagnostics["paths"]+=1
            dt=dt0
            for _ in range(steps):
                H0=acquisition.energy(x)+sum(v*v for v in p)/(2*mass)
                success=False
                for _retry in range(max_retries+1):
                    grad=acquisition.energy_gradient(x)
                    ph=[pi-.5*dt*gi for pi,gi in zip(p,grad)]
                    xn=[xi+dt*pi/mass for xi,pi in zip(x,ph)]
                    if problem.admissible(xn):
                        gn=acquisition.energy_gradient(xn)
                        pn=[pi-.5*dt*gi for pi,gi in zip(ph,gn)]
                        vn=acquisition.energy(xn)
                        H1=vn+sum(v*v for v in pn)/(2*mass)
                        if math.isfinite(H1) and abs(H1-H0)<=tolerance*max(1,abs(H0)):
                            success=True
                            break
                    diagnostics["rejected_steps"]+=1
                    dt*=.5
                if not success:
                    break
                # x is the potential trough before the newly ascending step.
                if len(values)>=2 and values[-2]>values[-1] and vn>values[-1]:
                    if depth<max_depth and diagnostics["paths"]+len(queue)<max_paths:
                        queue.append((list(x),depth+1))
                        diagnostics["branches"]+=1
                x,p=xn,pn
                values.append(vn)
                proposals.append(list(x))
                diagnostics["accepted_steps"]+=1
        return proposals,diagnostics

