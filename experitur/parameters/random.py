from typing import Any, Dict, List, Union

from experitur.core.parameters import ParameterGenerator, ParameterGeneratorIter
from experitur.helpers.merge_dicts import merge_dicts

try:
    from sklearn.model_selection import ParameterSampler

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


class _RandomSamplerIter(ParameterGeneratorIter):
    def __iter__(self):
        # Parameter names that are relevant in this context
        parameter_names = set(
            k
            for k in self.parameter_generator.param_distributions.keys()
            if k not in self.ignored_parameter_names
        )

        for parent_configuration in self.parent:
            # Firstly, produce existing configurations to give downstream samplers the chance to
            # produce missing sub-configurations.

            # Retrieve all trials that match parent_configuration
            existing_trials = self.experiment.ctx.store.match(
                func=self.experiment.func,
                parameters=parent_configuration.get("parameters", {}),
            )

            existing_params_set = set()
            for trial in existing_trials:
                existing_params_set.add(
                    tuple(
                        sorted(
                            (k, v)
                            for k, v in trial.data["parameters"].items()
                            if k in parameter_names
                        )
                    )
                )

            # Yield existing configurations
            for existing_params in existing_params_set:
                yield merge_dicts(
                    parent_configuration, parameters=dict(existing_params)
                )

            # Calculate n_iter as n_iter - already existing iterations
            n_iter = self.parameter_generator.n_iter - len(existing_params_set)

            for params in ParameterSampler(
                {
                    k: v
                    for k, v in self.parameter_generator.param_distributions.items()
                    if k not in self.ignored_parameter_names
                },
                n_iter,
            ):

                yield merge_dicts(parent_configuration, parameters=params)


class Random(ParameterGenerator):
    """
    Parameter sampler based on :py:class:`sklearn.model_selection.ParameterSampler`.

    If all parameters are presented as a list, sampling without replacement is performed.
    If at least one parameter is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous parameters.

    Parameters:
        param_distributions (dict): Dictionary with parameters names as keys and distributions or lists of values to try.
            Distributions must provide a rvs method for sampling (such as those from :py:mod:`scipy.stats`).
            If a list is given, it is sampled uniformly.
        n_iter (int): Number of parameter settings that are produced.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.parameters import Random
            from scipy.stats.distributions import expon

            @Random({"a": [1,2], "b": expon()}, 4)
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        This example will produce four runs, e.g. "1 0.898", "1 0.923", "2 1.878", and "2 1.038".

    """

    _iterator = _RandomSamplerIter
    _str_attr = ["n_iter"]

    def __init__(self, param_distributions: Dict[str, Union[List, Any]], n_iter: int):
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not available.")  # pragma: no cover

        self.param_distributions = param_distributions
        self.n_iter = n_iter

    @property
    def varying_parameters(self):
        return {
            k: v
            for k, v in self.param_distributions.items()
            if not isinstance(v, list) or len(v) > 1
        }

    @property
    def invariant_parameters(self):
        return {
            k: v
            for k, v in self.param_distributions.items()
            if isinstance(v, list) and len(v) <= 1
        }
