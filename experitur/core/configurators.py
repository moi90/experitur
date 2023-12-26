import collections.abc
import fnmatch
import itertools
import random
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from typing_extensions import final

from experitur.helpers.merge_dicts import merge_dicts
from experitur.util import unset


class BaseConfigurationSampler(metaclass=ABCMeta):
    """
    Base class for all configuration samplers.
    """

    def __iter__(self):
        return self.sample()

    @abstractmethod
    def sample(
        self, exclude: Optional[Set] = None
    ) -> Iterator[Mapping]:  # pragma: no cover
        """Sample trial configurations."""
        while False:
            yield {}

    def contains_subset_of(
        self, configuration: Mapping, exclude: Optional[Set] = None
    ) -> bool:  # pragma: no cover
        """
        Return True if there exists a sample that is a subset of `configuration`.

        - `configuration` matches if it contains additional keys not produced by the sampler.
        - `configuration` does not match if it lacks keys produced by the sampler.
        - `configuration` does not match if values for existing keys are different.
        """
        raise NotImplementedError(f"{type(self).__qualname__}.contains_subset_of")

    def contains_superset_of(self, configuration: Mapping) -> bool:  # pragma: no cover
        """
        Return True if there exists a sample that is a superset of `configuration`.

        - `configuration` does not match if it contains additional keys not produced by the sampler.
        - `configuration` matches  if it lacks keys produced by the sampler.
        - `configuration` does not match if values for existing keys are different.
        """
        raise NotImplementedError(f"{type(self).__qualname__}.contains_superset_of")


class _RootSampler(BaseConfigurationSampler):
    """Special sampler that produces an empty parameter choice."""

    def __init__(self) -> None:
        pass

    def sample(self, exclude=None) -> Iterator[Mapping]:
        yield {"parameters": {}}

    def contains_subset_of(
        self, configuration: Mapping, exclude: Optional[Set] = None
    ) -> bool:
        """Return True if there exists a sample that is a subset of `configuration`, i.e. always."""

        # print(f"{type(self).__qualname__}.contains_subset_of", configuration, exclude)

        return True

    def contains_superset_of(self, configuration: Mapping) -> bool:
        """Return True if there exists a sample that is a superset of `configuration`, i.e. if `configuration` is empty."""

        configuration = dict(configuration)

        # Pop parameters and check if empty
        parameters = configuration.pop("parameters", {})
        if parameters:
            return False

        # Check rest of configuration if empty
        return not configuration


class Configurable(metaclass=ABCMeta):
    """ABC for classes that implement `prepend_configurator`."""

    @abstractmethod
    def prepend_configurator(self, configurator: "BaseConfigurator") -> None:
        """
        Prepend a configurator.

        Used by BaseConfigurator.__call__.
        """


AnyConfigurable = TypeVar("AnyConfigurable", bound=Configurable)


class BaseConfigurator:
    @abstractmethod
    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:  # pragma: no cover
        pass

    @abstractproperty
    def parameter_values(self) -> Mapping[str, Container]:  # pragma: no cover
        """Information about the values that every parameter configured here can assume."""
        return {}

    def __add__(self, other) -> "AdditiveConfiguratorChain":
        if not isinstance(other, BaseConfigurator):  # pragma: no cover
            return NotImplemented

        return AdditiveConfiguratorChain(self, other)

    def __mul__(self, other) -> "MultiplicativeConfiguratorChain":
        if not isinstance(other, BaseConfigurator):  # pragma: no cover
            return NotImplemented

        return MultiplicativeConfiguratorChain(self, other)

    def __call__(self, configurable: AnyConfigurable) -> AnyConfigurable:
        """
        Prepend the Configurator to a Configurable.

        Allows a Configurator object to be used as a decorator.
        """
        configurable.prepend_configurator(self)
        return configurable


class ConfigurationSampler(BaseConfigurationSampler):
    """
    Configuration sampler base class.

    Contains default implementations for __init__, contains_subset_of, and contains_superset_of.
    """

    @final
    def __init__(
        self, configurator: "Configurator", parent: "BaseConfigurationSampler"
    ) -> None:
        self.configurator = configurator
        self.parent = parent

    def contains_subset_of(
        self, configuration: Mapping, exclude: Optional[Set] = None
    ) -> bool:
        """
        Return True if there exists a sample that is a subset of `configuration`.

        - `configuration` matches if it contains additional keys not produced by the sampler.
        - `configuration` does not match if it lacks keys produced by the sampler.
        - `configuration` does not match if values for existing keys are different.
        """

        if exclude is None:
            exclude = set()

        # The parameter values that should contain the supplied configuration, without `exclude`ed
        # If the values contain <unset>, the respective parameter of a trial can assume any value.
        parameter_values = {
            k: v
            for k, v in self.configurator.parameter_values.items()
            if (k not in exclude) and (unset not in v)
        }

        conf_parameters = configuration.get("parameters", {})

        # Check if all configured parameters are contained
        if any(
            (k not in conf_parameters) or (conf_parameters[k] not in v)
            for k, v in parameter_values.items()
        ):
            return False

        parent_params = {
            k: v for k, v in conf_parameters.items() if k not in parameter_values
        }

        # Let parents check the rest of the configuration
        exclude = exclude.union(self.configurator.parameter_values.keys())
        return self.parent.contains_subset_of(
            dict(configuration, parameters=parent_params), exclude=exclude
        )

    def contains_superset_of(self, configuration: Mapping) -> bool:
        """
        Return True if there exists a sample that is a superset of `configuration`.

        - `configuration` does not match if it contains additional keys not produced by the sampler (or its parents).
        - `configuration` matches if it lacks keys produced by the sampler.
        - `configuration` does not match if values for existing keys are different.
        """

        values = self.configurator.parameter_values

        own_params, parent_params = split_dict(
            configuration.get("parameters", {}), values
        )

        # Check parameters configured here
        if any(v not in values[k] for k, v in own_params.items()):
            return False

        # Let parents check the rest of the configuration
        return self.parent.contains_superset_of(
            dict(configuration, parameters=parent_params)
        )

    @staticmethod
    def prepare_values_exclude(values: Mapping, exclude: Optional[Set] = None):
        """Select the entries from `values` that are not excluded and update `excluded`."""
        if exclude is None:
            exclude = set()

        values = {k: v for k, v in values.items() if k not in exclude}

        exclude = exclude.union(values.keys())

        return values, exclude


class Configurator(BaseConfigurator):
    """
    Configurator base class with default implementation for build_sampler.

    Derived classes are expected to define __str_attrs__ and to contain a nested _Sampler.

    Example:

        .. code-block:: python

            class MyConfigurator(Configurator):
                __str_attrs__ = ("foo",)

                def __init__(self, foo):
                    self.foo = foo

                class _Sampler(ConfigurationSampler):
                    def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
                        for parent_configuration in self.parent.sample(exclude=exclude):
                            yield merge_dicts(
                                parent_configuration,
                                parameters={"foo": self.configurator.foo}
                            )
    """

    _Sampler: Type[ConfigurationSampler]

    __str_attrs__: Tuple[str, ...]

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        return self._Sampler(self, parent)

    def __str__(self):
        return "<{name} {parameters}>".format(
            name=self.__class__.__name__,
            parameters=", ".join(f"{k}={getattr(self, k)}" for k in self.__str_attrs__),
        )


def combine_parameter_values(v_left, v_right, drop_unset=False):
    if isinstance(v_left, tuple) and isinstance(v_right, tuple):
        if drop_unset:
            v_right = tuple(v for v in v_right if v is not unset)

        v_right = tuple(v for v in v_right if v not in v_left)
        return v_left + v_right

    raise NotImplementedError(
        f"extend_parameter_values not implemented for {v_left!r} + {v_right!r}"
    )  # pragma: no cover


def extend_parameter_values(
    left: Dict[str, Container], right: Mapping[str, Container]
) -> None:
    """Extend `left` with `right`."""

    for k, v_right in right.items():
        if k not in left:
            left[k] = v_right
            continue

        left[k] = combine_parameter_values(left[k], v_right)


class MultiplicativeConfiguratorChain(BaseConfigurator):
    """
    Multiplicative configurator chain.
    The result is the cross-product of all contained configurators.
    """

    def __init__(self, *configurators: BaseConfigurator) -> None:
        self.configurators = configurators

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        sampler = _RootSampler() if parent is None else parent

        for c in self.configurators:
            sampler = c.build_sampler(sampler)

        return sampler

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        parameter_values = ParameterSpace()
        for c in self.configurators:
            parameter_values.update_multiplicative(c.parameter_values)
        return parameter_values

    def __mul__(self, other) -> "MultiplicativeConfiguratorChain":
        if not isinstance(other, BaseConfigurator):  # pragma: no cover
            return NotImplemented

        return MultiplicativeConfiguratorChain(*self.configurators, other)

    def __str__(self):
        return "(" + (" * ".join(str(c) for c in self.configurators)) + ")"


class GenerativeContainer(collections.abc.Container):
    def __init__(self, values=None) -> None:
        super().__init__()

        self.values = []
        self.children = []

        if values is not None:
            self.update(values)

    def update(self, other):
        if isinstance(other, (list, tuple, set)):
            for elm in other:
                self.add(elm)
        elif isinstance(other, GenerativeContainer):
            for elm in other.values:
                self.add(elm)
            self.children.extend(other.children)
        else:
            self.children.append(other)

    def add(self, value):
        if value in self.values:
            return
        self.values.append(value)

    def discard(self, value):
        self.values = [v for v in self.values if v != value]

    def __contains__(self, x) -> bool:
        return x in self.values or any(x in c for c in self.children)

    def __str__(self) -> str:
        parts = [f"{v}" for v in self.values]
        for c in self.children:
            parts.append(f"*{c}")
        return "{" + (", ".join(parts)) + "}"

    def __repr__(self) -> str:
        return f"{type(self)}({self})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple, set)):
            if self.children:
                return False

            if len(self.values) != len(other):
                return False

            for elm in self.values:
                if elm not in other:
                    return False

            return True

        if isinstance(other, GenerativeContainer):
            if len(self.values) != len(other.values):
                return False

            for elm in self.values:
                if elm not in other.values:
                    return False

            if len(self.children) != len(other.children):
                return False

            for child in self.children:
                if child not in other.children:
                    return False

            return True

        return NotImplemented

    def is_invariant(self):
        return len(self.values) < 2 and not self.children


class ParameterSpace(Dict[str, GenerativeContainer]):
    def update(self, *args, **kwargs):
        raise NotImplementedError("Use update_additive or update_multiplicative")

    def update_additive(
        self, *parameter_spaces: Mapping[str, Container], **kwargs: Container
    ):
        parameter_spaces = parameter_spaces + (kwargs,)

        for p_space in parameter_spaces:
            if isinstance(p_space, _UnsetParameterSpace):
                continue

            for k, v in p_space.items():
                v = GenerativeContainer(v)
                if k not in self:
                    self[k] = v
                else:
                    self[k].update(v)

    def update_multiplicative(
        self, *parameter_spaces: Mapping[str, Container], **kwargs: Container
    ):
        parameter_spaces = parameter_spaces + (kwargs,)

        for p_space in parameter_spaces:
            if isinstance(p_space, _UnsetParameterSpace):
                for k in self.keys():
                    if k in p_space:
                        self[k] = GenerativeContainer((unset,))
                continue

            for k, v in p_space.items():
                v = GenerativeContainer(v)
                if k not in self:
                    self[k] = v
                else:
                    if unset not in v:
                        self[k] = v
                    else:
                        v.discard(unset)
                        self[k].update(v)


class AdditiveConfiguratorChain(BaseConfigurator):
    """
    Additive configurator chain.
    The result is the concatenation of all contained configurators.

    Args:
        *configurators: Child configurators.
        shuffle (bool, optional): Shuffle child configurations.
    """

    def __init__(self, *configurators: BaseConfigurator, shuffle=False) -> None:
        self.configurators = configurators
        self.shuffle = shuffle

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        return self._Sampler(  # pylint: disable=no-member
            tuple(c.build_sampler(parent) for c in self.configurators),
            shuffle=self.shuffle,
        )

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        parameter_values = ParameterSpace()

        # Update values
        for c in self.configurators:
            parameter_values.update_additive(c.parameter_values)

        # Add unset for all parameters that are missing in one of the child configurators
        for c in self.configurators:
            for missing_key in set(parameter_values.keys()) - (
                c.parameter_values.keys()
            ):
                parameter_values[missing_key].add(unset)

        return parameter_values

    def __add__(self, other) -> "BaseConfigurator":
        if not isinstance(other, BaseConfigurator):  # pragma: no cover
            return NotImplemented

        return AdditiveConfiguratorChain(*self.configurators, other)

    def __str__(self):
        return "(" + (" + ".join(str(c) for c in self.configurators)) + ")"

    class _Sampler(BaseConfigurationSampler):
        def __init__(
            self, samplers: Iterable[BaseConfigurationSampler], shuffle: bool
        ) -> None:
            self.samplers = samplers
            self.shuffle = shuffle

        def sample(self, exclude=None) -> Iterator[Mapping]:
            if self.shuffle:
                generators = [s.sample(exclude) for s in self.samplers]
                while generators:
                    g = random.choice(generators)
                    try:
                        yield next(g)
                    except StopIteration:
                        generators.remove(g)

            for s in self.samplers:
                yield from s.sample(exclude)  # pylint: disable=protected-access

        def contains_subset_of(
            self, configuration: Mapping, exclude: Optional[Set] = None
        ) -> bool:
            """Return True if there exists a sample that is a subset of `configuration`, i.e. if there is one in a child."""

            return any(
                s.contains_subset_of(configuration, exclude) for s in self.samplers
            )

        def contains_superset_of(self, configuration: Mapping) -> bool:
            """Return True if there exists a sample that is a superset of `configuration`, i.e. if there is one in a child."""

            return any(s.contains_superset_of(configuration) for s in self.samplers)


def split_dict(mapping: Mapping, indicator):
    result = ({}, {})
    for k, v in mapping.items():
        result[k not in indicator][k] = v

    return result


class Const(Configurator):
    """
    Constant parameter configurator.

    Parameters may be passed as a mapping and/or as keyword arguments.

    Parameters:
        values (Mapping): The parameters, as a dictionary mapping parameter names to values.
        **kwargs: Additional parameters.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.configurators import Const

            @Const({"a": 1, "b": 2}, c=3)
            @Experiment()
            def example1(parameters: Trial):
                print(parameters["a"], parameters["b"], parameters["c"])

        This example will produce "1 2 3".
    """

    __str_attrs__ = ("values",)

    def __init__(self, values: Optional[Mapping[str, Any]] = None, **kwargs):
        if values is None:
            self.values = kwargs
        else:
            self.values = {**values, **kwargs}

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {k: (v,) for k, v in self.values.items()}

    class _Sampler(ConfigurationSampler):
        configurator: "Const"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            values, exclude = self.prepare_values_exclude(
                self.configurator.values, exclude
            )

            for parent_configuration in self.parent.sample(exclude=exclude):
                # print(self.configurator, ":", values)
                yield merge_dicts(parent_configuration, parameters=values)


class ZeroConfigurator(Configurator):
    """
    Empty parameter configurator.

    This is different from Const() without arguments in that it DOES NOT YIELD ANY configurations.
    """

    __str_attrs__ = tuple()

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {}

    class _Sampler(ConfigurationSampler):
        configurator: "ZeroConfigurator"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            while False:
                yield {}


def parameter_product(p: Mapping[str, Iterable]):
    """Iterate over the points in the grid."""

    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            yield params


class Grid(Configurator):
    """
    Grid parameter generator that produces all parameter-value-combinations in the grid.

    Parameters:
        grid (Mapping): The parameter grid to explore, as a dictionary mapping parameter names to sequences of allowed values.
        shuffle (bool): If False, the combinations will be generated in a deterministic manner.
            *Deprecated!* Use RandomGrid instead.

    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.configurators import Grid

            @Grid({"a": [1,2], "b": [3,4]})
            @Experiment()
            def example1(parameters: Trial):
                print(parameters["a"], parameters["b"])

            @Experiment2(parameters={"a": [1,2], "b": [3,4]})
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        Both examples are equivalent and will produce "1 3", "1 4", "2 3", and "2 4".
    """

    __str_attrs__ = ("grid", "shuffle")

    shuffle = False

    def __init__(self, grid: Mapping[str, Iterable], shuffle: bool = False):
        if shuffle and not isinstance(self, RandomGrid):
            warnings.warn(
                "shuffle is deprecated, use RandomGrid instead.", DeprecationWarning
            )

        self._validate_grid(grid)

        self.grid = grid
        self.shuffle = shuffle

    def _validate_grid(self, grid: Mapping):
        for k, v in grid.items():
            if not isinstance(k, str):
                raise ValueError(f"Key {k!r} is not str")
            if not isinstance(v, collections.abc.Iterable):
                raise ValueError(f"Value {v!r} for parameter {k} is not Iterable")

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {k: tuple(v) for k, v in self.grid.items()}

    class _Sampler(ConfigurationSampler):
        configurator: "Grid"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            grid, exclude = self.prepare_values_exclude(self.configurator.grid, exclude)

            grid_product = list(parameter_product(grid))

            for parent_configuration in self.parent.sample(exclude=exclude):
                if self.configurator.shuffle:
                    random.shuffle(grid_product)

                for values in grid_product:
                    yield merge_dicts(parent_configuration, parameters=values)


class RandomGrid(Grid):
    """
    Grid parameter generator that produces all parameter-value-combinations in the grid in random order.
    """

    def __init__(self, grid: Mapping[str, Iterable]):
        super().__init__(grid, shuffle=True)

    shuffle = True


def validate_configurators(configurators) -> List[BaseConfigurator]:
    """
    Check configurators argument.

    Convert None, Mapping, Iterable to list of :py:class:`BaseConfigurator`s.
    """

    if configurators is None:
        return []
    if isinstance(configurators, Mapping):
        return [Grid(configurators)]
    if isinstance(configurators, Iterable) and not isinstance(
        configurators, (str, bytes)
    ):
        return sum((validate_configurators(c) for c in configurators), [])
    if isinstance(configurators, BaseConfigurator):
        return [configurators]

    raise ValueError(f"Unsupported type for configurators: {configurators!r}")


def is_invariant(configured_values: Any):
    """Return True if not more than one single value is configured."""

    try:
        is_invariant = configured_values.is_invariant
    except AttributeError:
        pass
    else:
        return is_invariant()

    if hasattr(configured_values, "__len__"):
        return len(configured_values) < 2

    return False


class FilterConfig(Configurator):
    """
    Filter configurations, e.g. to avoid invalid parameter combinations.

    Args:
    """

    def __init__(self, filter_func: Callable):
        self.filter_func = filter_func

    @property
    def parameter_values(self):
        return {}

    class _Sampler(ConfigurationSampler):
        configurator: "FilterConfig"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            for parent_configuration in self.parent.sample(exclude=exclude):
                parameters = parent_configuration.get("parameters", {})

                if not self.configurator.filter_func(parameters):
                    continue

                yield parent_configuration


class _UnsetParameterSpace(Mapping[str, Container]):
    def __init__(self, patterns: Sequence[str]) -> None:
        self.patterns = patterns

    def __contains__(self, key):
        return any(fnmatch.fnmatchcase(key, n) for n in self.patterns)

    def __getitem__(self, key: str) -> Container:
        if key in self:
            return {unset}
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError(repr(self) + ".__iter__")

    def __len__(self) -> int:
        raise NotImplementedError(repr(self) + ".__len__")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.patterns})"


class Clear(Configurator):
    """
    Clear parameters matching the given patterns.

    Args:
        *names (str): Parameter names to remove from the configuration.
    """

    __str_attrs__ = ("patterns",)

    def __init__(self, *patterns: str):
        self.patterns = patterns

    @property
    def parameter_values(self):
        return _UnsetParameterSpace(self.patterns)

    class _Sampler(ConfigurationSampler):
        configurator: "Clear"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            for parent_configuration in self.parent.sample(exclude=exclude):
                # Make a copy of the parent configuration
                parent_configuration = dict(parent_configuration)

                # Remove parameters that match any of the provided names
                parent_configuration["parameters"] = {
                    k: v
                    if not any(
                        fnmatch.fnmatchcase(k, n) for n in self.configurator.patterns
                    )
                    else unset
                    for k, v in parent_configuration.get("parameters", {}).items()
                }

                yield parent_configuration

        def contains_subset_of(
            self, configuration: Mapping, exclude: Optional[Set] = None
        ) -> bool:
            """
            Return True if there exists a sample that is a subset of `configuration`.

            A configuration matches if it does not contain any keys matched by the provided patterns
            """

            if exclude is None:
                exclude = set()

            conf_parameters = configuration.get("parameters", {})

            if any(
                fnmatch.fnmatchcase(k, n)
                for n in self.configurator.patterns
                for k, v in conf_parameters.items()
                if v != unset
            ):
                return False

            for k, v in conf_parameters.items():
                if any(fnmatch.fnmatchcase(k, n) for n in self.configurator.patterns):
                    exclude.add(k)

            print(f"Parents...")

            # Let parents check the rest of the configuration
            return self.parent.contains_subset_of(configuration, exclude=exclude)

        def contains_superset_of(self, configuration: Mapping) -> bool:
            """
            Return True if there exists a sample that is a superset of `configuration`.

            - `configuration` does not match if it contains additional keys not produced by the sampler (or its parents).
            - `configuration` matches if it lacks keys produced by the sampler.
            - `configuration` does not match if values for existing keys are different.
            """

            values = self.configurator.parameter_values

            own_params, parent_params = split_dict(
                configuration.get("parameters", {}), values
            )

            # Check parameters configured here
            if any(v not in values[k] for k, v in own_params.items()):
                return False

            # Let parents check the rest of the configuration
            return self.parent.contains_superset_of(
                dict(configuration, parameters=parent_params)
            )
