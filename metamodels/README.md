# Metamodel plugins

Use create_metamodel(config) with exactly one of plugin or factory.
Registered names: gp, svm, pce, bnn. External adapters use module:class factories.

## Common contract

Metamodel Protocol requires fit(X,Y), predict(x,gradient=False), capabilities,
snapshot, training_log. X has shape (n,d), Y has shape (n,), n>=2.
Y always contains raw scalar g values. Inputs and targets must be finite.

predict returns a frozen Prediction dataclass. Unsupported fields are None.
prediction.require("mean") raises a clear error when a field is unavailable.
Models return named fields; tuple unpacking is not supported.
New array models also provide predict_many(X), currently a loop over single inputs.

| Plugin | Prediction fields | Gradients when requested |
|---|---|---|
| GP | mean, epistemic_variance | mean_gradient, epistemic_variance_gradient |
| SVM | decision_function | decision_gradient |
| PCE | mean; optional ensemble_responses, ensemble_variance | mean_gradient |
| BNN | mean, epistemic_variance, observation_noise_variance, posterior_responses | mean_gradient, epistemic_variance_gradient |

Regression results are in original g units (variances in squared units).
Gradients are with respect to the input passed to predict, including the chain rule
through internal scaling. SVM decision scores are not g values or probabilities.
GP/SVM/BNN fit input scaling using training data only.

## GP / Kriging_GP.yaml

GaussianProcessRegressor with ConstantKernel times RBF. Removes the empirical
output mean and standardizes targets. This is GP regression, not GLS-estimated
ordinary/universal Kriging. Predictive variance is latent posterior variance;
no new observation noise is added.

Parameters: lengthscale (scalar or dimension-wise vector in scaled input units),
amplitude, nugget (scaled target squared units), optimize, optimizer_restarts, seed.
Kernel bounds are [1e-3,1e3]; optimize=false fixes kernel parameters.
Variance floor is 1e-12 with zero variance gradient when active.
Optimization warnings are emitted and saved in training_log.

## SVM / SVM.yaml

SVC learns failure=-1 for raw g<=0 and safe=+1 for g>0.
Parameters: C, kernel (rbf/linear), gamma (scale/auto/positive),
class_weight (balanced/null), seed. No posterior variance is fabricated.

Both failure and safe observations are required. A single-class fit raises a
clear error; this data-dependent condition cannot be checked before observing g.
The example uses linear_normal beta=0 to make both classes easier to obtain.
Its acquisition is minimum absolute decision score, a simple baseline rather
than a paper-specific ASVM policy.

## PCE / PCE.yaml

Total-degree orthonormal probabilists' Hermite basis for independent N(0,1)
coordinates. Least-squares/ridge fitting; the constant term is not penalized.
Parameters: degree, ridge, max_terms, bootstrap_members (0 or >=2), seed.
There is no sparse basis selection. Inputs are not rescaled using training data.
Other probability laws require a matching basis or explicit probability transform.

Pairs-bootstrap coefficients provide an ensemble proxy, not a GP posterior.
PCE does not declare epistemic_variance, so PCE + U is rejected.
The example uses absolute predicted g and budget-only termination.
Rank-deficient designs are reported as underdetermined=true in training_log.


## BNN / BNN.yaml

One tanh hidden layer and a linear output. Every weight and bias has a diagonal
Gaussian variational posterior. A Gaussian likelihood and prior are used.
Reparameterized expected negative log likelihood plus analytic KL(q||p) is
optimized with Adam; the objective is normalized per observation.
This small NumPy implementation is not a paper reproduction.

Parameters: hidden_units, epochs, learning_rate, prior_std, noise_std,
initial_weight_std, train_samples, posterior_samples (>=2), seed.
noise_std is the likelihood standard deviation in STANDARDIZED target units.
Every fit restarts from the seed, without warm start.
Posterior weight draws are fixed after fit, so Reference and NH use the same
function and gradients throughout an iteration.

epistemic_variance is the sample variance (ddof=1) of latent response draws.
observation_noise_variance is separate and is not used as learning uncertainty.
The example U criterion uses posterior mean/std as a heuristic; it does not
assert that the BNN response posterior is exactly Gaussian.
Pf integrates the sign of the posterior mean, not posterior failure probability.

## Preflight compatibility

pipeline.prepare checks before true g calls or output directory creation:

- Required model methods and state attributes.
- Acquisition.required_capabilities against model.capabilities.
- Evaluator required_capabilities and required_any_capabilities.
- Acquisition kind supported by the evaluation stopping heuristic.
- NH energy/energy_gradient capabilities, callable methods and required model gradients.
- PCE input distribution and GP lengthscale dimension.

Protocols describe the contract; preflight checks declarations and callables.
It cannot prove that external plugins implement their declared capabilities
correctly or predict accurately. Numerical plugin tests verify those properties.

An acquisition declares required_capabilities, gradient_required_capabilities,
capabilities, kind, direction and implements score/energy/energy_gradient as needed.
The runner also uses score_points, energy_points, gradient_points counters.
An evaluator declares required_capabilities, optionally required_any_capabilities
and supported_acquisition_kinds.

## Configuration and reproducibility

Each configs/Method_Name.yaml is a complete standalone run specification.
There is no hidden YAML inheritance. Copy a file for each paper/methodology,
then set the model, learning policy and compatible sampling/evaluation adapters.
profile=metamodel_demo denotes a working method demo, not reproduction of P01-P30.
The four example YAMLs are not a controlled model comparison: in particular SVM
uses a different beta. Match problem, DOE, seeds, budgets and evaluation for comparisons.

evaluation.jsonl includes model_snapshot, model_capabilities, training_log.
effective_config.json records the run configuration.
snapshot hashes config, training data and fitted state; it is not a serialized model.
Dependency/platform changes can change fitting and snapshot values.

Run tests:
    python -m unittest discover -s tests -v

Tests cover central-difference gradients, GP/SVM backend agreement, polynomial
recovery, BNN ELBO gradients and learning, frozen posterior replay, incompatible
combinations before g, and all four YAMLs in Reference/NH modes.

## Implementation references

- [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [NumPy HermiteE basis](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermevander.html)
- [Blundell et al., Weight Uncertainty in Neural Network (2015)](https://proceedings.mlr.press/v37/blundell15.html)



## Paper P01 ordinary Kriging factory

`factory: metamodels.ordinary_kriging:OrdinaryKriging` selects the P01 equations
(constant GLS trend, anisotropic Gaussian correlation, profile MLE and BLUP MSE,
including trend-estimation uncertainty). Parameters: theta (positive scalar/vector),
theta_bounds ([lower, upper]), nugget (nonnegative numerical regularization),
optimize (boolean), optimizer_restarts, batch_size and seed. Coordinates and targets
use sample-standard-deviation scaling. theta is in those scaled coordinates.
DACE-style boxmin uses multiplicative coordinate and pattern moves. The last accepted
theta starts the next fit; configured theta and geometric restarts provide alternatives.
Only valid finite fitted states are adopted. The warm start, restarts and nugget mean
an identical MATLAB DACE trajectory is not promised. The existing registry name gp
continues to select KrigingGP.
`predict_moments(X)` is an optional batched mean/epistemic-variance API, used by
population evaluation; the required single-input Prediction contract is unchanged.
