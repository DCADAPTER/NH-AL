# Adaptive-learning framework

One method YAML selects the problem, metamodel, acquisition, selector, optional
NH refiner and evaluation policy. The execution path is:

    YAML -> pipeline.prepare -> capability checks -> initial true g evaluations
         -> model.fit -> evaluation -> Reference selection -> optional NH
         -> selected true g evaluation -> repeat / save results

## File responsibilities

| Path | Responsibility |
|---|---|
| run.py | CLI arguments, method configuration selection and paper lookup |
| pipeline.py | Construct and validate components, run the AL loop, save results |
| metamodels/ | Model registry, prediction contract and GP/SVM/PCE/BNN implementations |
| g.py | Problem definition and logged/cached true g evaluations |
| adaptive.py | Acquisition, Reference selection and refinement acceptance |
| nh.py | Optional NH prototype |
| evaluation.py | Surrogate Pf estimation and stop-plugin routing (legacy demo evaluators retained) |
| stop/ | Method-specific convergence decisions, without true g calls |
| configs/ | One complete YAML per paper or methodology |
| tests/test_metamodels.py | Model fitting, prediction contracts and numerical gradients |
| tests/test_adaptive.py | Acquisition gradients and acceptance rules |
| tests/test_pipeline.py | CLI/config integration, preflight checks and paired-run invariants |
| papers.yaml | P01-P30 literature specifications |

Runtime code does not import the tests. All model implementations live in
metamodels/; the runner has no built-in duplicate GP implementation.
The default configuration is configs/Kriging_GP.yaml.

## Install and run

Python 3.10+ is required. Use the project's virtual environment.

~~~powershell
python -m pip install -r requirements.txt
python run.py --check-config
python run.py --list-metamodels
python run.py --config configs/AK-MCS-U.yaml --mode paired --output runs/gp_trial_01
python run.py --config configs/SVM.yaml --output runs/svm_trial_01
python run.py --list-papers
python run.py --paper P16
python -m unittest discover -s tests -v
~~~

For library use:

~~~python
from pipeline import execute, read_config

cfg = read_config("configs/PCE.yaml")
summaries = execute(cfg, "paired", "runs/pce_trial_01")
print(summaries["reference"]["final"])
print(summaries["nh"]["final"])
~~~

prepare returns a Pipeline with named components rather than an opaque tuple.
execute validates every requested mode before any true g call. In paired mode,
an incompatible NH configuration prevents the Reference run from consuming budget.
Existing output directories are never overwritten.

## Repeated experiments and resume

Set `repetitions: 10` in the method config (AK-MCS-U default). Run:

```powershell
python run.py --config configs/AK-MCS-U.yaml --mode paired --output runs/repeated_01
python run.py --config configs/AK-MCS-U.yaml --mode paired --output runs/repeated_01 --resume
```

Outputs are `rep_001/reference`, `rep_001/nh`, etc.; `results.json` collects completed
summaries. Point Analyze.ipynb's `EXPERIMENT_DIR` at the experiment root to aggregate
completed pairs, compare paired differences, and cache reliability metrics. Each repetition
adds 100,000 to the base RNG and initial-design seeds; both modes share the same
initial design and populations within a repetition. Explicit metamodel seeds also
advance. An `initial_design.input_file` intentionally keeps the saved design fixed.

`--resume` skips each completed mode. Interrupted modes replay fitting/selection
from the start using previously logged successful g values, then continue with new
g calls. This preserves theta warm starts and avoids recomputing recorded g values.
Replay checks coordinates and fails if the trajectory changes. Use the same code
and environment. A g call interrupted before its result was logged must be retried.
Old incomplete folders are retained as `.MODE_interrupted_*`, and a durable replay
file protects observations if replay itself is interrupted. No run data is deleted.
Timing fields on resumed runs describe the rebuilding attempt (true_seconds includes
reused observation times); they are not the total wall time across interruptions.

Increase `repetitions` and rerun with `--resume` to append repetitions. Other config
changes are rejected; choose a fresh output folder for a different experiment.
Resume is for experiment folders created by this runner, not older standalone runs.
Run only one process per output folder. Configs without `repetitions` retain the
original single-run output layout unless `--resume` is explicitly used.

## Method configurations

| YAML | Model | Acquisition / stopping |
|---|---|---|
| configs/Kriging_GP.yaml | RBF GP with kernel fitting | U / audit U and MC COV |
| configs/AK-MCS-U.yaml | Ordinary Kriging | U / `stop.AK-MCS-U:Stop` |
| configs/AK-MCS-EFF.yaml | Ordinary Kriging | EFF / `stop.AK-MCS-EFF:Stop` |
| configs/AL-SPCE.yaml | Stochastic SPCE | AL-SPCE / weighted K-medoids batch selection |
| configs/SVM.yaml | RBF SVC | absolute decision margin / call budget |
| configs/PCE.yaml | Hermite PCE with bootstrap | absolute predicted g / call budget |
| configs/BNN.yaml | Variational BNN | posterior mean/std U / audit U and MC COV |

Each YAML is standalone, without implicit inheritance.
`configs/AK-MCS-EFF.yaml` uses the paper's reliability settings `a = 0` and
`epsilon = 2 sigma`, maximizes EFF, and stops learning when
`max(EFF) <= 0.001`; it also requires the configured MC coefficient-of-variation
criterion. The U and EFF methods use separate stop plugins.
Use profile=metamodel_demo for these implementations. P01-P30 entries in
papers.yaml remain literature specifications; the available method configs are
not verified paper reproductions.
The JSON-formatted YAML examples do not need a YAML parser. Block YAML uses PyYAML.

`configs/AL-SPCE.yaml` is an executable stochastic demonstration of P26, not a
reproduction of its application cases. It fits a stochastic PCE with a uniform
latent response variable, maximizes `f_X(x) Var_ep[s_hat(x)]`, and selects up to
five candidates per iteration using weighted K-medoids. `max_true_calls` is the
only stopping rule: the paper identifies a robust learning-convergence threshold
as unresolved. Batches are truncated to the remaining true-call budget. The
example uses a small stochastic linear problem; the implementation currently
requires independent standard-normal physical inputs and one uniform latent
variable. On later fits, noise calibration searches five log-spaced candidates
within +/-5% of the previous estimate; the initial fit estimates noise jointly.
Use `metamodel.plugin: spce` or `metamodel.factory` to construct the model.
Run the example with:

```powershell
python run.py --config configs/AL-SPCE.yaml --output runs/al_spce_demo
```

SVM needs both failure and safe observations; its example uses beta=0 while the
other examples use beta=2.5. Align problems, initial designs, seeds, budgets and
evaluation policies before comparing model performance.
PCE assumes independent standard-normal coordinates. Its bootstrap variance is
an ensemble proxy and is not exposed as GP posterior variance.
BNN has one tanh hidden layer, a variational Gaussian weight posterior and frozen
posterior draws between fits. Observation noise and weight uncertainty are separate.
Audit-based stopping is a heuristic, not a global accuracy certificate.

## Contracts and experiment records

All models implement Metamodel and return Prediction. Unsupported fields are None.
Choose metamodel.plugin (gp/svm/pce/bnn) or an external metamodel.factory
(module:class). See [model contracts and settings](metamodels/README.md).

NH is an additional optimization step after Reference selection. Only the final
selected point calls true g; the shadow Reference does not consume extra calls.
Original acquisition score decides acceptance; ties retain the Reference.
The NH implementation is a prototype that can be replaced through nh.factory.

Outputs: g_calls.jsonl, selection.jsonl, evaluation.jsonl, fixed_samples.json,
training.csv, effective_config.json, summary.json.
Evaluation records include model capabilities, fitted-state identity and training
logs, including GP optimizer warnings and PCE design rank.
Judge NH by true g calls, final Pf error and total elapsed time.
System/batch/noisy/time-dependent adapters still require their additional contracts.

The literature and research design are documented in
[the four-module reference document](AL_30_Papers_Four_Module_NH_20261015.md).



## AK-MCS-U / modified Rastrigin (P01)

```powershell
python run.py --config configs/AK-MCS-U.yaml --check-config
python run.py --config configs/AK-MCS-U.yaml --mode paired --output runs/rastrigin_pair_01
```

The source PDF supplied by the user was checked directly: Section 2, p.147,
Eqs.(6)-(11); Section 3.2-3.3, pp.147-150; Section 4.2, pp.151-152, Eq.(18),
Table 4. DOI: https://doi.org/10.1016/j.strusafe.2011.01.002.

The problem is `g(x)=10-sum(x_i^2-5*cos(2*pi*x_i))`, with two independent standard
normal inputs and failure at g<=0. The selected `OrdinaryKriging` factory implements
constant GLS trend, anisotropic Gaussian correlation, profile maximum-likelihood
estimation and the ordinary-Kriging prediction MSE including trend uncertainty.
It also supplies analytic input gradients for NH and chunked population predictions.
See `metamodels/ordinary_kriging.py`. The existing `gp` plugin remains the earlier
empirical-mean sklearn GP; the paper configuration explicitly chooses OrdinaryKriging.

The Python theta search follows DACE's boxmin coordinate and pattern moves. It also
uses the last accepted theta as the next fit's starting point, tries configured and
geometrically spaced restart points, and rejects invalid fitted states. The article does not specify
theta initial values/bounds, warm starts or nugget. Those numerical choices are
exposed/documented in YAML or implementation notes, not attributed to the paper.
Input/target sample-standardization follows the DACE manual:
https://omicron.dk/dace/dace.pdf. A small configured nugget regularizes the matrix;
latent MSE is floored at 1e-12 for U and NH numerical stability.

### Initial points and reproducible NH comparison

`initial_design.size` controls the point count (default 10).
`method: optimal_lhd` builds a centered Latin hypercube, then optimizes its minimum
pairwise Euclidean distance in the unit probability cube using multistart coordinate
exchange. Each column preserves one point in each stratum. The inverse normal CDF
maps it to standard-normal input coordinates. This is a finite heuristic search,
not a guarantee of the globally optimal LHD or maximin distances in physical space.
`seed`, `restarts`, and `iterations` are configurable.

Each run saves `initial_design.json` before any true g call, then adds the initial
responses as `Y`. The record includes `X`, unit-cube coordinates, seed and optimization
settings. Paired runs explicitly copy the same coordinates and independently evaluate
the same initial points so both call budgets include the initial evaluations.
`initial_design.input_file` replays the saved X coordinates even with a different seed.
Paths are relative to the working directory (or absolute). The configured size must
match the saved file. A ready-to-use 10-point design is stored at
`designs/rastrigin_olhd10_seed20261026.json`; input_file defaults to null so changing
size regenerates a design automatically. Reuse the same full config/seed to also keep
MC populations and model fitting identical at initialization.

### Population and stopping

The current comparison config uses two independent standard-normal MC sets:
`sampling.population: independent` fixes 60,000 adaptive candidates (`pool_size`,
seed + 55); `evaluation.integration_size` fixes 60,000 evaluation points (seed + 22).
Reference selection and NH starting points use only the adaptive set. Pf, CoV and
minimum U use only the evaluation set. Paired runs copy identical coordinates for
both sets without sharing mutable lists. Random-from-population initial designs
also use the adaptive set; the default Optimal LHD remains independently generated.

`evaluation.stop.population_mode: holdout` selects this comparison protocol.
It is a change from the paper's shared S, not a paper reproduction. To restore
shared-population behavior set **both** `sampling.population: integration` and
`evaluation.stop.population_mode: shared`. Old configs without population_mode
default to shared. Other demo configs retain their per-iteration candidate pools.

Only the evaluation set grows, by 60,000, after min U >= 2 while MC CoV exceeds .05
(or is undefined because no failures are predicted). `max_population_size` is
2,000,000. Expansion uses seed + 44, so paired runs share the same expansion
prefix even if they expand at different iterations. The adaptive set stays fixed.
In shared mode, the expanded evaluation set also remains the candidate set.
`adaptive:BatchedPoolSelector` preserves min-U selection and distance exclusion;
predictions run in bounded chunks (`metamodel.batch_size`).

`evaluation.stop.factory: stop.AK-MCS-U:Stop` loads `stop/AK-MCS-U.py` via importlib.
The plugin declares `required_capabilities`, `supported_acquisition_kinds`, and
`requires_shared_population` / `requires_independent_population`, and implements
`evaluate(model, acquisition, population, metrics)`. The result contains `stop`,
`converged`, `stop_reason` and optionally `expand_population`.
`evaluation.PluginMCEvaluation` computes Pf, routes to the plugin and enforces caps.
Unknown plugins or incompatible combinations fail before any true g call.

Learning stops at min U >= 2 over the evaluation set. Shared mode excludes observed
points from the U check (they remain in Pf and CoV); holdout mode scores every
evaluation point. Overall convergence also requires MC CoV <= .05. Holdout
convergence is labeled `converged_holdout_u`; shared convergence retains
`converged_ak_mcs_u`. The stop plugin owns this decision, independently of NH.
An evaluation set used repeatedly for stopping is a validation set, not an untouched
final test set. Minimum U remains a model-confidence criterion, not a measured
true-g error or a guarantee of Pf accuracy. Splitting sets does not guarantee that
NH will improve or that either method will converge within the budget.
When learning is complete but CoV is too large, MC points are added and rechecked
without refitting or true g calls. Zero predicted failures cannot imply convergence.
`budget_exhausted` and `mc_population_limit` are distinct nonconverged outcomes.
`fixed_samples.json` stores the initial evaluation set as `integration`, the fixed
candidate set as `adaptive` in independent mode, and the legacy `audit` array.
`mc_population.json` contains the latest enlarged evaluation set when expansion
occurs. Evaluation logs include `population_mode`, population size and stop factory.
Existing runs and their effective configs retain their original shared semantics.

The intentional user change to the paper's initial design is Optimal LHD with 10
points instead of random points from S. The initial MC population now matches the
Rastrigin example's 60,000; its later growth increment and cap remain framework
settings. Table 4 reports Pf=.0734 and 416 calls for AK-MCS-U; these are historical
comparison values, not exact targets for the changed experiment. NH is a separate
prototype extension. No full numerical reproduction is claimed.
The default CLI config remains Kriging_GP.yaml; select AK-MCS-U.yaml explicitly.


## Hamiltonian parent with L-BFGS-B children

The AK-MCS-U config selects `nh.child.method: lbfgsb`. The root path always uses
Hamiltonian leapfrog. At a discrete energy trough, the parent queues a child and
continues its own trajectory; children run afterward on the same fitted model.
Each child minimizes `acquisition.energy` (U squared for AK-MCS-U), with analytic
input gradients. Final selection still uses the original U score and rejects
points too close to training points. Only the selected point calls true g.

`nh.child.radius` defines a box of +/- radius around the branching position in
input coordinates, intersected with the problem's symmetric search bound. For
standard-normal Rastrigin, radius=.5 means half a standard deviation per axis.
This is a user-tunable local search region, not a step-size setting. General
non-box domain violations terminate the child; arbitrary nonlinear constraints
are not handled by L-BFGS-B.

`maxiter`, `maxfun`, `maxls`, `ftol`, and `gtol` control each child. A wrapper enforces
maxfun even within a line search. Finite evaluated candidates are retained if
optimization reaches a limit; convergence status does not replace original-score
acceptance. Parent numerical tolerances and dt do not control local-solver children.
L-BFGS-B children are terminal (no grandchildren); max_paths counts the root and
all children, and max_depth=0 disables branching. No trough means no child.

`nh.child.method: hamiltonian` restores Hamiltonian children (the default when the
child block is omitted). Both methods share the root trajectory and branching rule
for the same starting point/model/random seed. Adam remains unimplemented and is
rejected explicitly. The selection log's nh_diagnostics records method, path counts
and per-L-BFGS-B-child start/bounds, best energy, status, evaluation counts and time.
All energy/gradient calls also contribute to the existing sampling cost counters.
