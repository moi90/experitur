from experitur.core.parameters import ParameterGenerator, ParameterGeneratorIter
from experitur.helpers.merge_dicts import merge_dicts


class _PruneIter(ParameterGeneratorIter):
    parameter_generator: "Prune"

    def __iter__(self):

        for parent_configuration in self.parent:
            parameters = parent_configuration.get("parameters")

            pruning_config = merge_dicts(
                self.parameter_generator.config, parameters=parameters
            )

            yield merge_dicts(parent_configuration, pruning_config=pruning_config)


class Prune(ParameterGenerator):
    """
    Record the parameter configuration so far to enable pruning among groups of trials.

    Args:
        quantile: Keep the best x% trials.
    """

    _iterator = _PruneIter

    def __init__(
        self,
        step_name: str,
        quantile: float,
        maximize=None,
        minimize=None,
        check_every_n=1,
        min_steps=0,
        min_count=5,
    ):
        if maximize is not None and minimize is not None:
            raise ValueError("maximize and minimize are mutually exclusive")

        invert_signs = False
        if maximize is not None:
            minimize = maximize
            invert_signs = True

        if minimize is None:
            raise ValueError("Neither maximize nor minimize was set.")

        self.config = {
            "step_name": step_name,
            "quantile": quantile,
            "minimize": minimize,
            "invert_signs": invert_signs,
            "check_every_n": check_every_n,
            "min_steps": min_steps,
            "min_count": min_count,
        }

    @property
    def independent_parameters(self):
        return {}
