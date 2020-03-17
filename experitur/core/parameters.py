import random
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import MutableMapping
from itertools import product
from typing import TYPE_CHECKING, Dict, Generator, List, Mapping, Optional

from sklearn.model_selection import ParameterGrid, ParameterSampler

from experitur.core import trial as _trial
from experitur.helpers.merge_dicts import merge_dicts

if TYPE_CHECKING:
    from experitur.core.experiment import Experiment


class SamplerIter(ABC):
    def __init__(
        self, sampler: "Sampler", experiment: "Experiment", parent: "SamplerIter",
    ):
        self.sampler = sampler
        self.experiment: "Experiment" = experiment
        self.parent = parent
        self.ignored_parameter_names = set()

    @abstractmethod
    def __iter__(self) -> Generator[_trial.Trial, None, None]:  # pragma: no cover
        for parent_configuration in self.parent:
            while False:
                yield merge_dicts(parent_configuration, parameters={...: ...})

    def ignoreParameters(self, parameter_names):
        self.parent.ignoreParameters(parameter_names)
        self.ignored_parameter_names.update(parameter_names)


class _NullSamplerIter(SamplerIter):
    """Sampler iterator that yields one empty parameter configuration."""

    def __init__(self):
        pass

    def __iter__(self):
        yield {}

    def ignoreParameters(self, parameter_names):
        pass


class Sampler(ABC):
    _iterator: SamplerIter
    _str_attr: List[str] = []

    def sample(self, experiment: "Experiment", parent: Optional[SamplerIter] = None):
        """
        Return a SamplerIter to sample parameter configurations.

        Parameters:
                experiment (Experiment): Experiment instance.
                parent (SamplerIter): Parent SamplerIter.

        Custom Samplers
        ---------------

        Subclasses have to implement `_iterator` and `parameters`.

        .. code-block:: python

            class MySamplerIter(SamplerIter):
                def __iter__(self):
                    for parent_configuration in self.parent:
                        for x in range(3):
                            yield {**parent_configuration, "x": x}

            class MySampler(Sampler):
                _iterator = MySamplerIter
            
        """

        if parent is None:
            parent = _NullSamplerIter()

        parent.ignoreParameters(
            set(self.varying_parameters.keys()) | set(self.invariant_parameters.keys())
        )
        return self._iterator(self, experiment, parent)

    @abstractproperty
    def varying_parameters(self) -> Mapping:  # pragma: no cover
        """Parameters in this sampler. Does not include parameters that do not vary."""
        pass

    @property
    def invariant_parameters(self) -> Mapping:
        """Invariant parameters in this sampler."""
        return {}

    def __str__(self):
        return "<{name} {parameters}>".format(
            name=self.__class__.__name__,
            parameters=", ".join(f"{k}={getattr(self, k)}" for k in self._str_attr),
        )


def parameter_product(p):
    """Iterate over the points in the grid."""

    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params


class _GridSamplerIter(SamplerIter):
    def __iter__(self):

        params_list = list(
            parameter_product(
                {
                    k: v
                    for k, v in self.sampler.parameter_grid.items()
                    if k not in self.ignored_parameter_names
                }
            )
        )

        for parent_configuration in self.parent:
            if self.sampler.shuffle:
                random.shuffle(params_list)

            for params in params_list:
                print(parent_configuration)
                yield merge_dicts(parent_configuration, parameters=params)


class GridSampler(Sampler):
    _iterator = _GridSamplerIter
    _str_attr = []

    def __init__(self, parameter_grid={}, shuffle=False):
        self.parameter_grid = parameter_grid
        self.shuffle = shuffle

    @property
    def varying_parameters(self):
        return {k: v for k, v in self.parameter_grid.items() if len(v) > 1}

    @property
    def invariant_parameters(self):
        return {k: v for k, v in self.parameter_grid.items() if len(v) <= 1}


class _RandomSamplerIter(SamplerIter):
    def __iter__(self):
        # Parameter names that are relevant in this context
        parameter_names = set(
            k
            for k in self.sampler.param_distributions.keys()
            if k not in self.ignored_parameter_names
        )

        for parent_configuration in self.parent:
            # Firstly, produce existing configurations to give downstream samplers the chance to
            # produce missing sub-configurations.

            # Retrieve all trials that match parent_configuration
            existing_trials = self.experiment.ctx.store.match(
                callable=self.experiment.callable,
                parameters=parent_configuration.get("parameters", {}),
            )

            existing_params_set = set()
            for trial in existing_trials.values():
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
            n_iter = self.sampler.n_iter - len(existing_params_set)

            for params in ParameterSampler(
                {
                    k: v
                    for k, v in self.sampler.param_distributions.items()
                    if k not in self.ignored_parameter_names
                },
                n_iter,
            ):

                yield merge_dicts(parent_configuration, parameters=params)


class RandomSampler(Sampler):
    """
    Parameter sampler based on py:class:`sklearn.model_selection.ParameterSampler`.
    """

    _iterator = _RandomSamplerIter
    _str_attr = ["n_iter"]

    def __init__(self, param_distributions: Dict, n_iter):
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


class MultiSampler(Sampler):
    _str_attr = ["samplers"]

    def __init__(self, samplers=None):
        self.samplers: List[Sampler] = []

        if samplers is not None:
            self.addMulti(samplers)

    def add(self, sampler):
        self.samplers.append(sampler)

    def addMulti(self, samplers):
        for s in samplers:
            self.add(s)

    def sample(self, experiment, parent=None):
        if parent is None:
            parent = _NullSamplerIter()

        last_sampler_iter = parent
        for sampler in self.samplers:
            last_sampler_iter = sampler.sample(experiment, last_sampler_iter)

        return last_sampler_iter

    @property
    def varying_parameters(self) -> Dict:
        parameters = {}
        for sampler in self.samplers:
            parameters.update(sampler.varying_parameters)

            # Delete invariant parameters
            for ip in sampler.invariant_parameters:
                parameters.pop(ip, None)

        return parameters
