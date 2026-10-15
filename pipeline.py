"""Connect configured components and execute the adaptive-learning loop."""
import copy
import csv
import importlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from adaptive import accept_refinement
from g import Evaluator
from metamodels import Metamodel, create_metamodel, validate_pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "Kriging_GP.yaml"


def read_config(path):
    """Read one standalone YAML/JSON configuration."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("For block-style YAML install PyYAML") from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def factory(spec, *args):
    """Construct a non-metamodel adapter from an explicit module:class factory."""
    module, name = spec["factory"].split(":")
    return getattr(importlib.import_module(module), name)(*args)


@dataclass
class Pipeline:
    """A validated set of components for one Reference or NH run."""
    config: dict
    mode: str
    problem: object
    model: Metamodel
    selector: object
    refiner: object | None
    acquisition: object
    evaluator: object
    integration: list
    audit: list
    preparation_seconds: float

    def run(self, out):
        """Execute this validated combination; only the oracle calls true g."""
        start_run=time.perf_counter()
        cfg, mode = self.config, self.mode
        problem, model = self.problem, self.model
        selector, refiner = self.selector, self.refiner
        evaluator = self.evaluator
        integration, audit = self.integration, self.audit
        out = Path(out)
        out.mkdir(parents=True,exist_ok=False)
        (out/"effective_config.json").write_text(json.dumps(dict(cfg,mode=mode),indent=2),encoding="utf-8")
        seed=int(cfg["seed"])
        oracle=Evaluator(problem,out/"g_calls.jsonl")
        initial=int(cfg["initial_design"]["size"])
        X=problem.sample(random.Random(seed+11),initial,search=True)
        Y=[oracle.evaluate(x,0,"initial") for x in X]
        records=[]
        totals={"fit_seconds":0.0,"evaluation_seconds":0.0,"candidate_seconds":0.0,"reference_seconds":0.0,
                "nh_seconds":0.0,"rescore_seconds":0.0,"sampling_score_points":0,
                "sampling_energy_points":0,"sampling_gradient_points":0}
        # Write actual shared integration/audit inputs for reproducibility.
        (out/"fixed_samples.json").write_text(json.dumps({"integration":integration,"audit":audit}),encoding="utf-8")
        iteration=0
        while True:
            t=time.perf_counter()
            model.fit(X,Y)
            fit_seconds=time.perf_counter()-t
            totals["fit_seconds"]+=fit_seconds
            # Bind to the new fitted snapshot, then freeze across Reference/NH.
            self.acquisition = factory(cfg["acquisition"], model)
            acquisition = self.acquisition
            validate_pipeline(model, acquisition, evaluator, refiner)
            metrics=evaluator.evaluate(model,acquisition,oracle.n_calls)
            totals["evaluation_seconds"]+=metrics["evaluation_seconds"]
            metrics.update(iteration=iteration,n_true_calls=oracle.n_calls,model_snapshot=model.snapshot,
                           model_capabilities=sorted(model.capabilities),training_log=dict(model.training_log))
            with (out/"evaluation.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(metrics,allow_nan=False)+"\n")
            if metrics["stop"]:
                break
            # Independent RNG streams mean NH draws cannot change reference pools.
            t=time.perf_counter()
            candidates=problem.sample(random.Random(seed+10000+iteration),cfg["sampling"]["pool_size"],search=True)
            candidate_seconds=time.perf_counter()-t
            t=time.perf_counter()
            before=(acquisition.score_points,acquisition.energy_points,acquisition.gradient_points)
            x_ref,a_ref=selector.select(candidates,acquisition,X,problem)
            reference_seconds=time.perf_counter()-t
            x_selected,a_selected=list(x_ref),a_ref
            nh_seconds=rescore_seconds=0.0
            diagnostics={}
            if mode=="nh":
                t=time.perf_counter()
                proposals,diagnostics=refiner.refine(x_ref,acquisition,problem,random.Random(seed+20000+iteration))
                nh_seconds=time.perf_counter()-t
                t=time.perf_counter()
                x_selected,a_selected=accept_refinement(x_ref,a_ref,proposals,acquisition,
                                                       lambda x:selector.valid(x,X,problem))
                rescore_seconds=time.perf_counter()-t
            delta=(a_ref-a_selected) if acquisition.direction=="min" else (a_selected-a_ref)
            if delta < -1e-12:
                raise AssertionError("Original acquisition worsened")
            after=(acquisition.score_points,acquisition.energy_points,acquisition.gradient_points)
            rec={"iteration":iteration,"snapshot":model.snapshot,"mode":mode,"x_reference":x_ref,
                 "x_selected":x_selected,"a_reference":a_ref,"a_selected":a_selected,
                 "acquisition_gain":delta,"candidate_seconds":candidate_seconds,"reference_seconds":reference_seconds,"nh_seconds":nh_seconds,
                 "rescore_seconds":rescore_seconds,
                 "sampling_seconds":candidate_seconds+reference_seconds+nh_seconds+rescore_seconds,
                 "score_points":after[0]-before[0],"energy_points":after[1]-before[1],
                 "gradient_points":after[2]-before[2],"nh_diagnostics":diagnostics}
            with (out/"selection.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(rec,allow_nan=False)+"\n")
            records.append(rec)
            for key in ("candidate_seconds","reference_seconds","nh_seconds","rescore_seconds"):
                totals[key]+=rec[key]
            for key in ("score_points","energy_points","gradient_points"):
                totals["sampling_"+key]+=rec[key]
            # Only selected point calls g. Shadow reference is not evaluated.
            y=oracle.evaluate(x_selected,iteration+1,"selected")
            X.append(list(x_selected));Y.append(y)
            iteration+=1
        exact=problem.exact_pf()
        summary={"profile":cfg["profile"],"mode":mode,"target":problem.target,
                 "implementation":"demonstration_not_paper_reproduction",
                 "nh_implementation":"prototype_replace_with_verified_user_module" if mode=="nh" else None,
                 "n_true_calls":oracle.n_calls,"n_selected":len(records),"final":metrics,"exact_pf":exact,
                 "abs_pf_error":abs(metrics["pf_surrogate"]-exact) if exact is not None else None,
                 "true_seconds":oracle.total_seconds,**totals,"wall_seconds":time.perf_counter()-start_run+self.preparation_seconds}
        (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
        with (out/"training.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f);writer.writerow([f"x{i}" for i in range(problem.d)]+["g"])
            writer.writerows([*x,y] for x,y in zip(X,Y))
        print(json.dumps({"output":str(out),"mode":mode,"n_calls":oracle.n_calls,
                          "stop":metrics["stop_reason"],"pf":metrics["pf_surrogate"]}))
        return summary

def prepare(cfg, mode):
    """Build and validate a combination before output creation or any true g call."""
    start = time.perf_counter()
    cfg = copy.deepcopy(cfg)
    if cfg["profile"] != "metamodel_demo":
        raise NotImplementedError("P01-P30 remain specifications. Use metamodel_demo.")
    if mode not in ("reference", "nh"):
        raise ValueError("Unsupported mode")
    seed = int(cfg["seed"])
    problem = factory(cfg["problem"], cfg["problem"])
    model_cfg = dict(cfg["metamodel"])
    model_cfg.setdefault("seed", seed)
    cfg["metamodel"] = model_cfg
    model = create_metamodel(model_cfg)
    selector = factory(cfg["sampling"], cfg["sampling"])
    refiner = factory(cfg["nh"], cfg["nh"]) if mode == "nh" else None
    initial = int(cfg["initial_design"]["size"])
    limit = int(cfg["evaluation"]["max_true_calls"])
    if initial < 2 or limit < initial:
        raise ValueError("Require 2 <= initial size <= max_true_calls")
    if min(cfg["sampling"]["pool_size"], cfg["evaluation"]["integration_size"], cfg["evaluation"]["audit_size"]) < 1:
        raise ValueError("Pool sizes must be positive")
    acquisition = factory(cfg["acquisition"], model)
    integration = problem.sample(random.Random(seed+22), cfg["evaluation"]["integration_size"])
    audit = problem.sample(random.Random(seed+33), cfg["evaluation"]["audit_size"], search=True)
    evaluator = factory(cfg["evaluation"], cfg["evaluation"], integration, audit)
    validate_pipeline(model, acquisition, evaluator, refiner, problem)
    for name in ("select", "valid"):
        if not callable(getattr(selector, name, None)):
            raise ValueError(f"Sampling adapter must implement {name}()")
    return Pipeline(cfg, mode, problem, model, selector, refiner, acquisition,
                    evaluator, integration, audit, time.perf_counter() - start)

def prepare_runs(cfg, mode):
    """Validate every requested mode before any run can consume true g budget."""
    if mode not in ("reference", "nh", "paired"):
        raise ValueError("Unsupported mode")
    modes = ("reference", "nh") if mode == "paired" else (mode,)
    return {name: prepare(cfg, name) for name in modes}


def run(cfg, mode, out):
    """Programmatic entry point for a single run."""
    return prepare(cfg, mode).run(out)


def execute(cfg, mode, out):
    """Execute a single run or a paired comparison after complete preflight."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output directory; runs are never overwritten")
    pipelines = prepare_runs(cfg, mode)
    return {name: pipeline.run(out/name if mode == "paired" else out)
            for name, pipeline in pipelines.items()}
