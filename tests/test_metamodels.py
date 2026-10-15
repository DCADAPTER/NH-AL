"""Unit tests for plugin contracts, fitting, predictions and numerical gradients."""
import unittest
import numpy as np

from metamodels import Prediction, Metamodel, CompatibilityError, create_metamodel
from metamodels.bnn import BayesianNeuralNetwork


class MetamodelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.random.default_rng(12).normal(size=(60, 2))
        cls.Y = .3 + .8*cls.X[:, 0] - .4*cls.X[:, 1] + .1*cls.X[:, 0]**2
        cls.models = {}
        settings = {
            "gp": {"optimize": False, "lengthscale": 1.3},
            "svm": {"C": 100},
            "pce": {"degree": 2, "ridge": 0, "bootstrap_members": 12},
            "bnn": {"hidden_units": 12, "epochs": 700, "posterior_samples": 48},
        }
        for name, cfg in settings.items():
            model = create_metamodel(dict(cfg, plugin=name, seed=7))
            model.fit(cls.X, cls.Y)
            cls.models[name] = model

    def test_common_contract_and_honest_optional_outputs(self):
        for name, model in self.models.items():
            with self.subTest(name=name):
                self.assertIsInstance(model, Metamodel)
                result = model.predict([.2, -.3])
                self.assertIsInstance(result, Prediction)
                self.assertTrue(model.snapshot)
                self.assertEqual(model.training_log["n_samples"], len(self.X))
                self.assertEqual(result, model.predict([.2, -.3]))
                for capability in model.capabilities - {"input_gradient"}:
                    if not capability.endswith("gradient"):
                        self.assertIsNotNone(getattr(result, capability))
        svm = self.models["svm"].predict([0, 0])
        self.assertIsNone(svm.mean)
        self.assertIsNone(svm.epistemic_variance)
        pce = self.models["pce"].predict([0, 0])
        self.assertIsNone(pce.epistemic_variance)
        self.assertEqual(len(pce.ensemble_responses), 12)
        self.assertGreaterEqual(pce.ensemble_variance, 0)
        bnn = self.models["bnn"].predict([0, 0])
        self.assertAlmostEqual(bnn.epistemic_variance, np.var(bnn.posterior_responses, ddof=1))
        self.assertGreater(bnn.observation_noise_variance, 0)

    def test_predict_requires_fit_and_valid_dimensions(self):
        for name, model in self.models.items():
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    create_metamodel({"plugin": name}).predict([0, 0])
                for x in ([0], [0, 0, 0], [float("nan"), 0]):
                    with self.assertRaises(ValueError):
                        model.predict(x)
                for X, Y in (([[0, 0]], [1]), ([[0, 0], [1, 1]], [1]), ([[0, 0], [1, 1]], [1, float("nan")])):
                    with self.assertRaises(ValueError):
                        create_metamodel({"plugin": name}).fit(X, Y)

    def test_input_gradients_against_central_difference(self):
        pairs = (("mean", "mean_gradient"),
                 ("epistemic_variance", "epistemic_variance_gradient"),
                 ("decision_function", "decision_gradient"))
        for name, model in self.models.items():
            for point in ([.25, -.35], [-.4, .6]):
                x = np.asarray(point)
                result = model.predict(x, gradient=True)
                for value_field, grad_field in pairs:
                    if grad_field not in model.capabilities:
                        continue
                    for j in range(2):
                        with self.subTest(model=name, field=grad_field, axis=j):
                            offset = np.eye(2)[j] * 1e-5
                            numerical = (getattr(model.predict(x+offset), value_field) -
                                         getattr(model.predict(x-offset), value_field)) / 2e-5
                            actual = getattr(result, grad_field)[j]
                            self.assertAlmostEqual(actual, numerical, delta=2e-5*max(1, abs(numerical)))

    def test_gp_matches_backend_mean_and_latent_variance(self):
        model = self.models["gp"]
        x = np.array([.25, -.4])
        z = ((x-model.x_center)/model.x_scale)[None, :]
        mu, std = model.estimator.predict(z, return_std=True)
        result = model.predict(x)
        self.assertAlmostEqual(result.mean, model.y_center + model.y_scale*mu[0], places=10)
        self.assertAlmostEqual(result.epistemic_variance, model.y_scale**2*std[0]**2, places=10)

    def test_gp_optimized_kernel(self):
        model = create_metamodel({"plugin": "gp", "optimize": True})
        model.fit(self.X[:20], self.Y[:20])
        self.assertTrue(np.isfinite(model.predict([.1, .1]).mean))
        self.assertIn("kernel", model.training_log)

    def test_svm_orientation_and_backend_agreement(self):
        model = self.models["svm"]
        predicted = [model.predict(x).decision_function <= 0 for x in self.X]
        self.assertGreater(np.mean(predicted == (self.Y <= 0)), .95)
        x = np.array([.2, -.6])
        expected = model.estimator.decision_function(((x-model.x_center)/model.x_scale)[None, :])[0]
        self.assertAlmostEqual(model.predict(x).decision_function, expected, places=10)
        with self.assertRaisesRegex(ValueError, "both failure"):
            create_metamodel({"plugin": "svm"}).fit([[0], [1]], [1, 2])

    def test_linear_svm_gradient(self):
        model = create_metamodel({"plugin": "svm", "kernel": "linear"})
        model.fit([[-2], [-1], [1], [2]], [-2, -1, 1, 2])
        result = model.predict([.25], gradient=True)
        numeric = (model.predict([.25001]).decision_function-model.predict([.24999]).decision_function)/.00002
        self.assertAlmostEqual(result.decision_gradient[0], numeric, places=7)

    def test_pce_recovers_polynomial_on_unseen_points(self):
        model = self.models["pce"]
        x = np.random.default_rng(99).normal(size=(20, 2))
        expected = .3 + .8*x[:, 0] - .4*x[:, 1] + .1*x[:, 0]**2
        actual = [model.predict(row).mean for row in x]
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        deterministic = create_metamodel({"plugin": "pce", "degree": 1})
        deterministic.fit(self.X, self.Y)
        self.assertNotIn("ensemble_responses", deterministic.capabilities)
        self.assertIsNone(deterministic.predict([0, 0]).ensemble_responses)

    def test_bnn_learns_and_replays_frozen_posterior(self):
        model = self.models["bnn"]
        x = np.random.default_rng(33).normal(scale=.7, size=(25, 2))
        expected = .3 + .8*x[:, 0] - .4*x[:, 1] + .1*x[:, 0]**2
        actual = np.asarray([model.predict(row).mean for row in x])
        self.assertLess(np.sqrt(np.mean((actual-expected)**2)), .3)
        original = model.predict([.2, -.4], gradient=True)
        snapshot = model.snapshot
        model.fit(self.X, self.Y)
        self.assertEqual(model.snapshot, snapshot)
        self.assertEqual(model.predict([.2, -.4], gradient=True), original)

    def test_bnn_elbo_gradient_including_kl(self):
        model = BayesianNeuralNetwork({"hidden_units": 3})
        model.dimension = 2
        rng = np.random.default_rng(8)
        size = 2*3+2*3+1
        mean = rng.normal(scale=.2, size=size)
        logs = np.full(size, -2.0)
        x, y = rng.normal(size=(5, 2)), rng.normal(size=5)
        eps = rng.normal(size=(3, size))
        _, gm, gs = model._objective_gradient(mean, logs, x, y, eps)
        for parameter, gradient in ((0, gm), (1, gs)):
            for j in range(size):
                args = [mean.copy(), logs.copy()]
                args[parameter][j] += 1e-6
                hi = model._objective_gradient(*args, x, y, eps)[0]
                args[parameter][j] -= 2e-6
                lo = model._objective_gradient(*args, x, y, eps)[0]
                self.assertAlmostEqual(gradient[j], (hi-lo)/2e-6, delta=1e-5)

    def test_configuration_validation(self):
        for cfg in ({"plugin":"unknown"}, {"plugin":"gp", "factory":"metamodels.gp:KrigingGP"},
                    {"plugin":"gp", "lengthscale":-1}, {"plugin":"svm","C":0},
                    {"plugin":"pce","bootstrap_members":1}, {"plugin":"bnn","noise_std":0}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                create_metamodel(cfg)

    def test_pce_rejects_wrong_input_law_before_g(self):
        class OtherProblem:
            d = 2
            input_distribution = "uniform"
        with self.assertRaises(CompatibilityError):
            self.models["pce"].validate_problem(OtherProblem())


if __name__ == "__main__":
    unittest.main(verbosity=2)

