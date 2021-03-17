from typing import Any, Dict, List, TYPE_CHECKING, Union

from scipy.stats import distributions

from experitur.core.parameters import (
    DynamicValues,
    ParameterGenerator,
    ParameterGeneratorIter,
)
from experitur.helpers.merge_dicts import merge_dicts
from unavailable_object import UnavailableObject, check_available


try:
    import sklearn.model_selection as sklearn_model_selection
except ImportError:
    if not TYPE_CHECKING:
        sklearn_model_selection = UnavailableObject("sklearn.model_selection")


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
            existing_trials = self.experiment.ctx.trials.match(
                func=self.experiment.func,
                parameters=parent_configuration.get("parameters", {}),
            )

            existing_params_set = set()
            for trial in existing_trials:
                existing_params_set.add(
                    tuple(
                        sorted(
                            (k, v)
                            for k, v in trial.parameters.items()
                            if k in parameter_names
                        )
                    )
                )

            if self.child is not None:
                # Yield existing configurations
                for existing_params in existing_params_set:
                    yield merge_dicts(
                        parent_configuration, parameters=dict(existing_params)
                    )

            # Calculate n_iter as n_iter - already existing iterations
            n_iter = self.parameter_generator.n_iter - len(existing_params_set)

            for params in sklearn_model_selection.ParameterSampler(
                {
                    k: v
                    for k, v in self.parameter_generator.param_distributions.items()
                    if k not in self.ignored_parameter_names
                },
                n_iter,
            ):
                existing_trials = self.experiment.ctx.trials.match(
                    func=self.experiment.func,
                    parameters=parent_configuration.get("parameters", {}),
                )

                n_existing = len(existing_trials)
                if n_existing >= self.parameter_generator.n_iter:
                    print(f"{self.parameter_generator}: {n_existing} existing trials.")
                    break

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

    AVAILABLE = not isinstance(sklearn_model_selection, UnavailableObject)

    _iterator = _RandomSamplerIter
    _str_attr = ["param_distributions", "n_iter"]

    @staticmethod
    def uniform(low, high):
        return distributions.uniform(low, high - low)

    def __init__(self, param_distributions: Dict[str, Union[List, Any]], n_iter: int):
        check_available(sklearn_model_selection)

        self.param_distributions = param_distributions
        self.n_iter = n_iter

    @property
    def independent_parameters(self):
        return {
            k: v if isinstance(v, list) else DynamicValues()
            for k, v in self.param_distributions.items()
        }
