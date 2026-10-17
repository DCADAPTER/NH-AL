"""Problem and true-model module. All true evaluations are logged here."""
import json
import math
import time
from pathlib import Path

class Problem:
    input_distribution = "standard_normal_independent"
    def __init__(self, cfg):
        self.kind = cfg["type"]
        self.d = int(cfg.get("dimension", 2))
        self.beta = float(cfg.get("beta", 2.5))
        bound = cfg.get("search_bound", 5.0)
        self.bound = math.inf if bound is None else float(bound)
        if self.d < 1 or self.bound <= 0:
            raise ValueError("dimension and search_bound must be positive")
        if self.kind not in ("linear_normal", "four_branch", "rastrigin"):
            raise ValueError("Unknown problem: " + self.kind)
        if self.kind in ("four_branch", "rastrigin") and self.d != 2:
            raise ValueError(f"{self.kind} requires dimension=2")
        self.target = "P(g(X)<=0), X~N(0,I); search bound does not truncate integration"

    def sample(self, rng, n, search=False):
        result = []
        attempts = 0
        while len(result) < n:
            x = [rng.gauss(0, 1) for _ in range(self.d)]
            attempts += 1
            if not search or self.admissible(x):
                result.append(x)
            if attempts > max(100000, 10000*n):
                raise RuntimeError("Search-domain rejection sampling exhausted")
        return result

    def admissible(self, x):
        return len(x) == self.d and all(math.isfinite(v) and abs(v) <= self.bound for v in x)

    def g(self, x):
        if self.kind == "linear_normal":
            return self.beta - sum(x) / math.sqrt(self.d)
        if self.kind == "rastrigin":
            return 10 - sum(v*v - 5*math.cos(2*math.pi*v) for v in x)
        a, b = x
        return min(3 + .1*(a-b)**2 - (a+b)/math.sqrt(2),
                   3 + .1*(a-b)**2 + (a+b)/math.sqrt(2),
                   (a-b) + 6/math.sqrt(2), (b-a) + 6/math.sqrt(2))

    def exact_pf(self):
        if self.kind == "linear_normal":
            return .5*math.erfc(self.beta/math.sqrt(2))
        return None

class Evaluator:
    def __init__(self, problem, output):
        self.problem = problem
        self.output = Path(output)
        self.cache = {}
        self.n_calls = 0
        self.total_seconds = 0.0

    def evaluate(self, x, iteration, role):
        key = tuple(x)
        if key in self.cache:
            return self.cache[key]
        start = time.perf_counter()
        record = {"iteration": iteration, "role": role, "x": list(x)}
        try:
            y = float(self.problem.g(x))
            if not math.isfinite(y):
                raise ValueError("Non-finite true-model output")
            record.update(y=y, status="ok")
        except Exception as exc:
            record.update(y=None, status="failed", error=repr(exc))
            raise
        finally:
            elapsed = time.perf_counter()-start
            self.n_calls += 1
            self.total_seconds += elapsed
            record.update(seconds=elapsed, cumulative_calls=self.n_calls)
            with self.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False)+"\n")
        self.cache[key] = y
        return y

