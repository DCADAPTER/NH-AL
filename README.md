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
| evaluation.py | Surrogate Pf estimation and stopping policies |
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
python run.py --config configs/Kriging_GP.yaml --mode paired --output runs/gp_trial_01
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

## Method configurations

| YAML | Model | Acquisition / stopping |
|---|---|---|
| configs/Kriging_GP.yaml | RBF GP with kernel fitting | U / audit U and MC COV |
| configs/SVM.yaml | RBF SVC | absolute decision margin / call budget |
| configs/PCE.yaml | Hermite PCE with bootstrap | absolute predicted g / call budget |
| configs/BNN.yaml | Variational BNN | posterior mean/std U / audit U and MC COV |

Each YAML is standalone, without implicit inheritance.
Use profile=metamodel_demo for these implementations. P01-P30 remain literature
specifications; they are not implemented paper reproductions.
The JSON-formatted YAML examples do not need a YAML parser. Block YAML uses PyYAML.

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

