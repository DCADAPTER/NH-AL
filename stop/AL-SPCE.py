"""AL-SPCE has no paper-defined numerical convergence threshold."""


class Stop:
    required_capabilities = frozenset({"mean", "epistemic_variance"})
    supported_acquisition_kinds = frozenset({"al_spce"})
    requires_shared_population = False
    requires_independent_population = False

    def __init__(self, cfg):
        self.factory = "stop.AL-SPCE:Stop"

    def evaluate(self, model, acquisition, population, metrics):
        if not population:
            raise ValueError("AL-SPCE requires a nonempty fixed integration population")
        return {
            "learning_converged": False,
            "converged": False,
            "stop": False,
            "expand_population": False,
            "stop_reason": "continue",
        }
