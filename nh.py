"""Hamiltonian parent exploration with configurable child refinement.

This is a search prototype, not a distribution-preserving sampler. Children
start at discrete potential troughs. Local solvers are terminal children;
Hamiltonian children may branch further up to max_depth.
"""
import math
import time
from metamodels.base import integer, positive


class _ChildEvaluationLimit(Exception):
    pass


class _InvalidChildEvaluation(Exception):
    pass


class NestedHamiltonianPrototype:
    required_capabilities = frozenset({'energy', 'energy_gradient'})

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.mass = positive(cfg.get('mass', 1), 'nh.mass')
        self.dt = positive(cfg.get('dt', .03), 'nh.dt')
        self.steps = integer(cfg.get('steps', 15), 'nh.steps')
        self.max_paths = integer(cfg.get('max_paths', 4), 'nh.max_paths')
        self.max_depth = integer(cfg.get('max_depth', 2), 'nh.max_depth', 0)
        self.max_retries = integer(cfg.get('max_retries', 6), 'nh.max_retries', 0)
        self.tolerance = positive(cfg.get('energy_tolerance', 1e-3), 'nh.energy_tolerance')
        self.temperature = positive(cfg.get('temperature', .2), 'nh.temperature')
        child = cfg.get('child', {})
        self.child_method = child.get('method', 'hamiltonian')
        if self.child_method not in ('hamiltonian', 'lbfgsb'):
            raise ValueError('nh.child.method must be hamiltonian or lbfgsb; Adam is not implemented yet')
        if self.child_method == 'lbfgsb':
            self.radius = positive(child.get('radius', .5), 'nh.child.radius')
            self.child_maxiter = integer(child.get('maxiter', 30), 'nh.child.maxiter')
            self.child_maxfun = integer(child.get('maxfun', 80), 'nh.child.maxfun')
            self.child_maxls = integer(child.get('maxls', 20), 'nh.child.maxls')
            self.child_ftol = positive(child.get('ftol', 1e-12), 'nh.child.ftol', allow_zero=True)
            self.child_gtol = positive(child.get('gtol', 1e-6), 'nh.child.gtol')

    def _lbfgsb_child(self, start, acquisition, problem):
        """Bounded minimization of acquisition.energy, without true g calls.

        Keep finite evaluated points even when the optimizer reaches a limit.
        Original acquisition scoring and training-point exclusion stay in pipeline.
        """
        import numpy as np
        from scipy.optimize import minimize
        started = time.perf_counter()
        center = np.asarray(start, dtype=float)
        bound = float(getattr(problem, 'bound', math.inf))
        lower = np.maximum(center-self.radius, -bound)
        upper = np.minimum(center+self.radius, bound)
        if not problem.admissible(center) or np.any(lower > upper):
            raise ValueError('L-BFGS-B child must start inside the problem domain')
        proposals = []
        energies = []
        evaluations = 0
        initial_energy = None

        def objective(x):
            nonlocal evaluations, initial_energy
            if evaluations >= self.child_maxfun:
                raise _ChildEvaluationLimit()
            if np.any(x < lower) or np.any(x > upper) or not problem.admissible(x):
                raise _InvalidChildEvaluation('optimizer point outside allowed region')
            evaluations += 1
            value = float(acquisition.energy(x))
            grad = np.asarray(acquisition.energy_gradient(x), dtype=float)
            if grad.shape != center.shape or not math.isfinite(value) or not np.isfinite(grad).all():
                raise _InvalidChildEvaluation('nonfinite energy or invalid gradient')
            if initial_energy is None:
                initial_energy = value
            proposals.append(x.tolist())
            energies.append(value)
            return value, grad

        # SciPy can exceed maxfun within a line search; objective enforces a hard cap.
        try:
            result = minimize(objective, center, jac=True, method='L-BFGS-B',
                              bounds=list(zip(lower, upper)),
                              options={'maxiter': self.child_maxiter, 'maxfun': self.child_maxfun,
                                       'maxls': self.child_maxls, 'ftol': self.child_ftol,
                                       'gtol': self.child_gtol})
            status = {'success': bool(result.success), 'status': int(result.status),
                      'message': str(result.message), 'nit': int(result.nit),
                      'x_final': result.x.tolist()}
        except _ChildEvaluationLimit:
            status = {'success': False, 'status': 'evaluation_limit',
                      'message': 'Child energy/gradient evaluation budget reached',
                      'nit': None, 'x_final': None}
        except _InvalidChildEvaluation as exc:
            status = {'success': False, 'status': 'invalid_evaluation',
                      'message': str(exc), 'nit': None, 'x_final': None}
        best = int(np.argmin(energies)) if energies else None
        return proposals, dict(status, method='lbfgsb', start=center.tolist(),
                               bounds=list(map(list, zip(lower.tolist(), upper.tolist()))),
                               nfev=evaluations, njev=evaluations,
                               initial_energy=initial_energy,
                               best_energy=energies[best] if best is not None else None,
                               best_x=proposals[best] if best is not None else None,
                               seconds=time.perf_counter()-started)

    def refine(self, x_ref, acquisition, problem, rng):
        queue = [(list(x_ref), 0)]
        proposals = []
        diagnostics = {'paths': 0, 'branches': 0, 'rejected_steps': 0, 'accepted_steps': 0,
                       'parent_method': 'hamiltonian', 'child_method': self.child_method,
                       'hamiltonian_paths': 0, 'local_search_paths': 0, 'children': []}
        while queue and diagnostics['paths'] < self.max_paths:
            x, depth = queue.pop(0)
            diagnostics['paths'] += 1
            if depth > 0 and self.child_method == 'lbfgsb':
                child_points, child_log = self._lbfgsb_child(x, acquisition, problem)
                proposals.extend(child_points)
                diagnostics['local_search_paths'] += 1
                diagnostics['children'].append(dict(child_log, depth=depth))
                continue
            diagnostics['hamiltonian_paths'] += 1
            p = [rng.gauss(0, math.sqrt(self.mass*self.temperature)) for _ in x]
            values = [acquisition.energy(x)]
            dt = self.dt
            for _ in range(self.steps):
                h0 = acquisition.energy(x) + sum(v*v for v in p)/(2*self.mass)
                success = False
                for _retry in range(self.max_retries+1):
                    grad = acquisition.energy_gradient(x)
                    ph = [pi-.5*dt*gi for pi, gi in zip(p, grad)]
                    xn = [xi+dt*pi/self.mass for xi, pi in zip(x, ph)]
                    if problem.admissible(xn):
                        gn = acquisition.energy_gradient(xn)
                        pn = [pi-.5*dt*gi for pi, gi in zip(ph, gn)]
                        vn = acquisition.energy(xn)
                        h1 = vn + sum(v*v for v in pn)/(2*self.mass)
                        if math.isfinite(h1) and abs(h1-h0) <= self.tolerance*max(1, abs(h0)):
                            success = True
                            break
                    diagnostics['rejected_steps'] += 1
                    dt *= .5
                if not success:
                    break
                if len(values) >= 2 and values[-2] > values[-1] and vn > values[-1]:
                    if depth < self.max_depth and diagnostics['paths']+len(queue) < self.max_paths:
                        queue.append((list(x), depth+1))
                        diagnostics['branches'] += 1
                x, p = xn, pn
                values.append(vn)
                proposals.append(list(x))
                diagnostics['accepted_steps'] += 1
        return proposals, diagnostics
