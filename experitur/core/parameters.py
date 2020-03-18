import random
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import MutableMapping
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Set,
    Type,
)

from experitur.core import trial as _trial
from experitur.helpers.merge_dicts import merge_dicts

if TYPE_CHECKING:
    from experitur.core.experiment import Experiment


class ParameterGeneratorIter(ABC):
    def __init__(
        self,
        parameter_generator: "ParameterGenerator",
        experiment: "Experiment",
        parent: "ParameterGeneratorIter",
    ):
        self.parameter_generator = parameter_generator
        self.experiment: "Experiment" = experiment
        self.parent = parent
        self.ignored_parameter_names: Set[str] = set()

    @abstractmethod
    def __iter__(self) -> Generator[_trial.Trial, None, None]:  # pragma: no cover
        for parent_configuration in self.parent:
            while False:
                yield merge_dicts(parent_configuration, parameters={...: ...})

    def ignoreParameters(self, parameter_names):
        self.parent.ignoreParameters(parameter_names)
        self.ignored_parameter_names.update(parameter_names)


class _NullIter(ParameterGeneratorIter):
    """Sampler iterator that yields one empty parameter configuration."""

    def __init__(self):
        pass

    def __iter__(self):
        yield {}

    def ignoreParameters(self, parameter_names):
        pass


class ParameterGenerator(ABC):
    _iterator: Type[ParameterGeneratorIter]
    _str_attr: List[str] = []

    def generate(
        self, experiment: "Experiment", parent: Optional[ParameterGeneratorIter] = None,
    ):
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
            parent = _NullIter()

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

    def __call__(self, experiment: "Experiment"):
        experiment.add_parameter_generator(self, prepend=True)


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


class _GridIter(ParameterGeneratorIter):
    def __iter__(self):

        params_list = list(
            parameter_product(
                {
                    k: v
                    for k, v in self.parameter_generator.grid.items()
                    if k not in self.ignored_parameter_names
                }
            )
        )

        for parent_configuration in self.parent:
            if self.parameter_generator.shuffle:
                random.shuffle(params_list)

            for params in params_list:
                print(parent_configuration)
                yield merge_dicts(parent_configuration, parameters=params)


class Grid(ParameterGenerator):
    _iterator = _GridIter
    _str_attr: List[str] = []

    def __init__(self, grid={}, shuffle=False):
        self.grid = grid
        self.shuffle = shuffle

    @property
    def varying_parameters(self):
        return {k: v for k, v in self.grid.items() if len(v) > 1}

    @property
    def invariant_parameters(self):
        return {k: v for k, v in self.grid.items() if len(v) <= 1}


class Multi(ParameterGenerator):
    _str_attr = ["generators"]

    def __init__(self, parameter_generators=None):
        self.generators: List[ParameterGenerator] = []

        if parameter_generators is not None:
            self.addMulti(parameter_generators)

    def add(self, parameter_generator):
        self.generators.append(parameter_generator)

    def addMulti(self, samplers):
        for s in samplers:
            self.add(s)

    def generate(self, experiment, parent=None):
        if parent is None:
            parent = _NullIter()

        last_sampler_iter = parent
        for gen in self.generators:
            last_sampler_iter = gen.generate(experiment, last_sampler_iter)

        return last_sampler_iter

    @property
    def varying_parameters(self) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}
        for gen in self.generators:
            parameters.update(gen.varying_parameters)

            # Delete invariant parameters
            for ip in gen.invariant_parameters:
                parameters.pop(ip, None)

        return parameters
