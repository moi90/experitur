import collections.abc
import functools
import operator
import random
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

from experitur.core import trial as _trial
from experitur.helpers.merge_dicts import merge_dicts

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment


class DynamicValues:
    def __init__(self, n_total=None):
        self.n_total = n_total

    def __repr__(self):
        return f"<DynamicValues n_total={self.n_total}>"


def count_values(values):
    if isinstance(values, DynamicValues):
        return values.n_total

    return len(values)


class ParameterGeneratorIter(ABC):
    def __init__(
        self,
        parameter_generator: "ParameterGenerator",
        *,
        parent: "ParameterGeneratorIter",
    ):
        self.parameter_generator = parameter_generator
        self.parent = parent

        self.child: "Optional[ParameterGenerator]" = None

    @abstractmethod
    def __iter__(self) -> Generator[_trial.Trial, None, None]:  # pragma: no cover
        for parent_configuration in self.parent:
            while False:
                yield merge_dicts(parent_configuration, parameters={...: ...})

    @property
    def ignored_parameter_names(self):
        if self.child is None:
            return set()

        return set(self.child.independent_parameters.keys())


class _NullIter(ParameterGeneratorIter):
    """Sampler iterator that yields one empty parameter configuration."""

    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def __iter__(self):
        yield {}


class ParameterGenerator(ABC):
    _iterator: Type[ParameterGeneratorIter]
    _str_attr: List[str] = []

    def generate(self, *, parent: Optional[ParameterGeneratorIter] = None):
        """
        Return a ParameterGeneratorIter to sample parameter configurations.

        Parameters:
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

        parent.child = self

        return self._iterator(self, parent=parent)

    @abstractproperty
    def independent_parameters(
        self,
    ) -> Mapping[str, Union[List, DynamicValues]]:  # pragma: no cover
        """Independent parameters in this sampler."""

    def __str__(self):
        return "<{name} {parameters}>".format(
            name=self.__class__.__name__,
            parameters=", ".join(f"{k}={getattr(self, k)}" for k in self._str_attr),
        )

    def __call__(self, experiment: "Experiment"):
        experiment.add_parameter_generator(self, prepend=True)
        return experiment

    def __add__(self, other):
        return SequentialParameterGenerator([self, other])


class _SequentialParameterGeneratorIter(ParameterGeneratorIter):
    def __iter__(self):
        for sub_generator in self.parameter_generator.sub_generators:
            sub_generator: ParameterGenerator
            yield from sub_generator.generate(parent=self.parent)


def _collate_values(values: List[Union[List, DynamicValues]]):
    list_values = []
    dynamic_values = []

    for v in values:
        if isinstance(v, list):
            list_values.append(v)
        elif isinstance(v, DynamicValues):
            dynamic_values.append(v)
        else:
            raise ValueError(f"unexpected value {v!r}. Expected list or DynamicValues")

    list_values = functools.reduce(operator.add, list_values) if list_values else []
    dynamic_values = (
        functools.reduce(operator.add, dynamic_values) if dynamic_values else []
    )

    if not dynamic_values:
        return list_values

    return dynamic_values + list_values


class SequentialParameterGenerator(ParameterGenerator):
    """
    Concatenate parameter generators.

    The result is the concatenation of individual configurations.
    """

    _iterator = _SequentialParameterGeneratorIter
    _str_attr = []

    def __init__(self, sub_generators: Iterable[ParameterGenerator]):
        self.sub_generators = sub_generators

    @property
    def independent_parameters(self) -> Mapping:
        independent_parameters = defaultdict(list)
        for g in self.sub_generators:
            for p, v in g.independent_parameters.items():
                independent_parameters[p].append(v)

        independent_parameters = {
            k: _collate_values(values) for k, values in independent_parameters.items()
        }

        return independent_parameters


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
            # Parameter names that are relevant in this context
            parameters = {
                k: v
                for k, v in self.parameter_generator.parameters.items()
                if k not in self.ignored_parameter_names
            }

            yield merge_dicts(parent_configuration, parameters=parameters)


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
    def independent_parameters(self):
        return {k: [v] for k, v in self.parameters.items()}


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
    def independent_parameters(self):
        return self.grid.copy()


class Multi(collections.abc.Iterable, ParameterGenerator):
    """
    Nest parameter generators.

    The result is the cross product of all configured values.
    """

    _str_attr = ["generators"]

    def __init__(self, parameter_generators=None):
        self.generators: List[ParameterGenerator] = []

        if parameter_generators is not None:
            self.addMulti(parameter_generators)

    def __iter__(self):
        return iter(self.generators)

    def add(self, parameter_generator):
        self.generators.append(parameter_generator)

    def addMulti(self, samplers):
        for s in samplers:
            self.add(s)

    def generate(self, parent=None):
        if parent is None:
            parent = _NullIter()

        last_sampler_iter = parent
        for gen in self.generators:
            last_sampler_iter = gen.generate(parent=last_sampler_iter)

        return last_sampler_iter

    @property
    def independent_parameters(self) -> Mapping[str, Union[List, DynamicValues]]:
        independent_parameters = {}
        for gen in self.generators:
            independent_parameters.update(gen.independent_parameters)

        return independent_parameters


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
