import random
from abc import ABC, abstractmethod, abstractproperty
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Type,
)

from experitur.core import trial as _trial
from experitur.helpers.merge_dicts import merge_dicts

if TYPE_CHECKING:  # pragma: no cover
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

    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def __iter__(self):
        yield {}

    def ignoreParameters(self, parameter_names):
        pass


class ParameterGenerator(ABC):
    _iterator: Type[ParameterGeneratorIter]
    _str_attr: List[str] = []

    def generate(
        self, experiment: "Experiment", parent: Optional[ParameterGeneratorIter] = None
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
        return experiment


def parameter_product(p: Mapping[str, Iterable]):
    """Iterate over the points in the grid."""

    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params


class _ConstIter(ParameterGeneratorIter):
    def __iter__(self):
        for parent_configuration in self.parent:
            yield merge_dicts(
                parent_configuration, parameters=self.parameter_generator.parameters
            )


class Const(ParameterGenerator):
    """
    Constant parameters.

    Parameters may be passed as a mapping and/or as keyword arguments.

    Parameters:
        parameters (Mapping): The parameters, as a dictionary mapping parameter names to values.
        **kwargs: Additional parameters.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.parameters import Const

            @Const({"a": 1, "b": 2}, c=3)
            @Experiment()
            def example1(parameters: Trial):
                print(parameters["a"], parameters["b"], parameters["c"])

        This example will produce "1 2 3".
    """

    _iterator = _ConstIter
    _str_attr: List[str] = ["parameters"]

    def __init__(self, parameters: Optional[Mapping[str, Any]] = None, **kwargs):
        if parameters is None:
            self.parameters = kwargs
        else:
            self.parameters = {**parameters, **kwargs}

    @property
    def varying_parameters(self):
        return {}

    @property
    def invariant_parameters(self):
        return {k: v for k, v in self.parameters.items()}


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
                yield merge_dicts(parent_configuration, parameters=params)


class Grid(ParameterGenerator):
    """
    Generate all parameter value combinations in the grid.

    Parameters:
        grid (Mapping): The parameter grid to explore, as a dictionary mapping parameter names to sequences of allowed values.
        shuffle (bool): If False, the combinations will be generated in a deterministic manner.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.parameters import Grid

            @Grid({"a": [1,2], "b": [3,4]})
            @Experiment()
            def example1(parameters: Trial):
                print(parameters["a"], parameters["b"])

            @Experiment2(parameters={"a": [1,2], "b": [3,4]})
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        Both examples are equivalent and will produce "1 3", "1 4", "2 3", and "2 4".
    """

    _iterator = _GridIter
    _str_attr: List[str] = ["grid", "shuffle"]

    def __init__(self, grid: Mapping[str, Iterable], shuffle: bool = False):
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

    @property
    def invariant_parameters(self) -> Dict[str, Any]:
        invariant_parameters: Dict[str, Any] = {}
        for gen in self.generators:
            invariant_parameters.update(gen.invariant_parameters)

            # Delete invariant parameters
            for ip in gen.varying_parameters:
                invariant_parameters.pop(ip, None)

        return invariant_parameters


def check_parameter_generators(parameters) -> List[ParameterGenerator]:
    """
    Check parameters argument.

    Convert None, Mapping, Iterable to list of :py:class:`ParameterGenerator`s.
    """

    if parameters is None:
        return []
    if isinstance(parameters, Mapping):
        return [Grid(parameters)]
    if isinstance(parameters, Iterable):
        return sum((check_parameter_generators(p) for p in parameters), [])
    if isinstance(parameters, ParameterGenerator):
        return [parameters]

    raise ValueError(f"Unsupported type for parameters: {parameters!r}")
