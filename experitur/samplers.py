from itertools import product
from typing import Optional, Mapping
from sklearn.model_selection import ParameterGrid, ParameterSampler
import random
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List

from experitur import context
from experitur.trial import TrialStore


class SamplerIter(ABC):
    def __init__(self, sampler, trial_store, parent: "SamplerIter"):
        self.sampler = sampler
        self.trial_store = trial_store
        self.parent = parent
        self.ignored_parameter_names = set()

    @abstractmethod
    def __iter__(self):  # pragma: no cover
        for parameter_configuration in self.parent:
            while False:
                yield None

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

    def sample(self, trial_store, parent: Optional[SamplerIter] = None):
        """
        Return a SamplerIter to sample parameter configurations.

        Parameters:
                trial_store (TrialStore): Trial store.
                parent (SamplerIter): Parent SamplerIter.
        """

        if parent is None:
            parent = _NullSamplerIter()

        parent.ignoreParameters(self.parameters.keys())
        return self._iterator(self, trial_store, parent)

    @abstractproperty
    def parameters(self) -> Mapping:  # pragma: no cover
        """Parameters in this sampler. Does not include parameters that do not vary."""
        pass

    @property
    def invariant_parameters(self):
        return {}


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
        configurations = list(
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
                random.shuffle(configurations)

            for configuration in configurations:
                configuration = {**parent_configuration, **configuration}

                # TODO: Skip existing trials
                if False:
                    continue

                yield configuration


class GridSampler(Sampler):
    _iterator = _GridSamplerIter

    def __init__(self, parameter_grid={}, shuffle=False):
        self.parameter_grid = parameter_grid
        self.shuffle = shuffle

    @property
    def parameters(self):
        return {k: v for k, v in self.parameter_grid.items() if len(v) > 1}

    @property
    def invariant_parameters(self):
        return {k: v for k, v in self.parameter_grid.items() if len(v) <= 1}


class _RandomSamplerIter(SamplerIter):
    def __iter__(self):
        for parent_configuration in self.parent:
            # TODO: Calculate n_iter as n_iter - already existing iterations
            n_iter = self.sampler.n_iter

            for configuration in ParameterSampler(
                self.sampler.param_distributions, n_iter
            ):
                configuration = {**parent_configuration, **configuration}

                # TODO: Skip existing trials
                if False:
                    continue

                yield configuration


class RandomSampler(Sampler):
    """
    Parameter sampler based on py:class:`sklearn.model_selection.ParameterSampler`.
    """

    _iterator = _RandomSamplerIter

    def __init__(self, param_distributions: Dict, n_iter):
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    @property
    def parameters(self):
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
    def __init__(self, samplers=None):
        self.samplers: List[Sampler] = []

        if samplers is not None:
            self.addMulti(samplers)

    def add(self, sampler):
        self.samplers.append(sampler)

    def addMulti(self, samplers):
        for s in samplers:
            self.add(s)

    def sample(self, trial_store, parent=None):
        if parent is None:
            parent = _NullSamplerIter()

        last_sampler_iter = parent
        for sampler in self.samplers:
            last_sampler_iter = sampler.sample(trial_store, last_sampler_iter)

        return last_sampler_iter

    @property
    def parameters(self) -> Dict:
        parameters = {}
        for sampler in self.samplers:
            parameters.update(sampler.parameters)

            # Delete invariant parameters
            for ip in sampler.invariant_parameters:
                del parameters[ip]

        return parameters
