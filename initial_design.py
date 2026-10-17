"""Reproducible initial designs and explicit saved-coordinate replay."""
import json
import random
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.spatial.distance import pdist
from scipy.stats import qmc
from metamodels.base import integer


def initial_design(cfg, problem, seed, population):
    n = integer(cfg['size'], 'initial_design.size', 2)
    design_seed = integer(cfg.get('seed', seed+11), 'initial_design.seed', 0)
    if cfg.get('input_file'):
        raw = json.loads(Path(cfg['input_file']).read_text(encoding='utf-8-sig'))
        points = np.asarray(raw['X'], dtype=float)
        metadata = {'method': 'saved_coordinates', 'input_file': str(cfg['input_file'])}
    else:
        method = cfg.get('method', 'normal_rejection_in_search_domain')
        metadata = {'method': method, 'seed': design_seed}
        if method == 'optimal_lhd':
            if problem.input_distribution != 'standard_normal_independent':
                raise ValueError('optimal_lhd currently requires independent standard normals')
            restarts = integer(cfg.get('restarts', 8), 'initial_design.restarts')
            iterations = integer(cfg.get('iterations', 2000), 'initial_design.iterations', 0)
            rng = np.random.default_rng(design_seed)
            best, best_distance = None, -1.
            # Centered LHD, coordinate-exchange maximin search in probability space.
            # A finite search gives a locally optimized design, not a global optimum certificate.
            for restart in range(restarts):
                u = qmc.LatinHypercube(d=problem.d, scramble=False, seed=design_seed+restart).random(n)
                distance = float(pdist(u).min())
                for _ in range(iterations):
                    a, b = rng.choice(n, 2, replace=False)
                    j = rng.integers(problem.d)
                    u[a,j], u[b,j] = u[b,j], u[a,j]
                    proposed = float(pdist(u).min())
                    if proposed >= distance:
                        distance = proposed
                    else:
                        u[a,j], u[b,j] = u[b,j], u[a,j]
                if distance > best_distance:
                    best, best_distance = u.copy(), distance
            points = ndtri(best)
            metadata.update(criterion='maximin_in_unit_probability_space',
                            optimizer='multistart_coordinate_exchange', restarts=restarts,
                            iterations=iterations, min_distance=best_distance,
                            unit_points=best.tolist(), transform='inverse_standard_normal_cdf')
        elif method == 'random_from_mc_population':
            points = np.asarray(random.Random(design_seed).sample(population, n))
        elif method == 'normal_rejection_in_search_domain':
            points = np.asarray(problem.sample(random.Random(design_seed), n, search=True))
        else:
            raise ValueError('Unsupported initial_design.method')
    if points.shape != (n, problem.d) or not np.isfinite(points).all():
        raise ValueError('Initial coordinates must be finite and match configured size and dimension')
    if len(np.unique(points, axis=0)) != n or not all(problem.admissible(x) for x in points):
        raise ValueError('Initial coordinates must be distinct and admissible')
    return points.tolist(), metadata
