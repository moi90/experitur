from typing import (
    Any,
    Container,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Union,
)

import sklearn.model_selection
from scipy.stats import distributions

from experitur import get_current_context
from experitur.core.configurators import ConfigurationSampler, Configurator
from experitur.helpers.merge_dicts import merge_dicts


class _DistWrapper(Container):
    def __init__(self, dist) -> None:
        self.dist = dist

    def __contains__(self, x) -> bool:
        if isinstance(self.dist, distributions.rv_frozen):
            return self.dist.a <= x <= self.dist.b

        raise ValueError(f"Can not check for containment of {x!r} in {self.dist!r}")

    def __repr__(self) -> str:
        return repr(self.dist)


class Random(Configurator):
    """
    Configurator based on :py:class:`sklearn.model_selection.ParameterSampler`.

    If all parameters are presented as a list, sampling without replacement is performed.
    If at least one parameter is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous parameters.

    Parameters:
        distributions (dict): Dictionary with parameters names as keys and distributions or lists of values to try.
            Distributions must provide a rvs method for sampling (such as those from :py:mod:`scipy.stats`).
            If a list is given, it is sampled uniformly.
        n_iter (int): Number of parameter settings that are produced.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.configurators import Random
            from scipy.stats.distributions import expon

            @Random({"a": [1,2], "b": expon()}, 4)
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        This example will produce four runs, e.g. "1 0.898", "1 0.923", "2 1.878", and "2 1.038".

    """

    _str_attr = ["distributions", "n_iter"]

    @staticmethod
    def uniform(low, high):
        return distributions.uniform(low, high - low)

    def __init__(self, distributions: Dict[str, Union[List, Any]], n_iter: int):
        self.distributions = distributions
        self.n_iter = n_iter

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {
            k: tuple(v) if isinstance(v, Iterable) else _DistWrapper(v)
            for k, v in self.distributions.items()
        }

    class _Sampler(ConfigurationSampler):
        configurator: "Random"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:

            distributions, exclude = self.prepare_values_exclude(
                self.configurator.distributions, exclude
            )

            ctx = get_current_context()
            experiment = ctx.current_experiment

            for parent_configuration in self.parent.sample(exclude=exclude):
                # Firstly, produce existing configurations to give downstream samplers the chance to
                # produce missing sub-configurations.

                # Retrieve all trials that match parent_configuration
                existing_trials = ctx.trials.match(
                    func=experiment.func,
                    parameters=parent_configuration.get("parameters", {}),
                )

                existing_params_set = set()
                for trial in existing_trials:
                    existing_params_set.add(
                        tuple(
                            sorted(
                                (k, v)
                                for k, v in trial.parameters.items()
                                if k in distributions
                            )
                        )
                    )

                # Yield existing configurations
                for existing_params in existing_params_set:
                    yield merge_dicts(
                        parent_configuration, parameters=dict(existing_params)
                    )

                # Calculate n_iter as n_iter - already existing iterations
                n_iter = self.configurator.n_iter - len(existing_params_set)

                for params in sklearn.model_selection.ParameterSampler(
                    distributions, n_iter
                ):
                    existing_trials = ctx.trials.match(
                        func=experiment.func,
                        parameters=parent_configuration.get("parameters", {}),
                    )

                    n_existing = len(existing_trials)
                    if n_existing >= self.configurator.n_iter:
                        print(f"{self.configurator}: {n_existing} existing trials.")
                        break

                    yield merge_dicts(parent_configuration, parameters=params)
