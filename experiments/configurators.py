from abc import abstractmethod, abstractproperty
import abc
from experitur.helpers.merge_dicts import merge_dicts
from typing import (
    Any,
    Container,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Set,
    Type,
)
from typing_extensions import final
import itertools


class BaseConfigurationSampler(abc.ABC):
    """
    Base class for all configuration samplers.
    """

    def __iter__(self):
        return self.sample()

    @abstractmethod
    def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
        """Sample trial configurations."""
        while False:
            yield {}

    def contains_subset_of(
        self, configuration: Mapping, exclude: Optional[Set] = None
    ) -> bool:
        """
        Return True if there exists a sample that is a subset of `configuration`.

        - `configuration` matches if it contains additional keys not produced by the sampler.
        - `configuration` does not match if it lacks keys produced by the sampler.
        - `configuration` does not match if values for existing keys are different.
        """
        raise NotImplementedError(f"{type(self).__qualname__}.contains_subset_of")

    def contains_superset_of(self, configuration: Mapping) -> bool:
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


class BaseConfigurator:
    @abstractmethod
    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        pass

    @abstractproperty
    def parameter_values(self) -> Mapping[str, Container]:
        """Information about the values that every parameter configured here can assume."""
        return {}

    def __add__(self, other) -> "BaseConfigurator":
        if not isinstance(other, BaseConfigurator):
            return NotImplemented

        return AdditiveConfiguratorChain(self, other)

    def __mul__(self, other) -> "BaseConfigurator":
        if not isinstance(other, BaseConfigurator):
            return NotImplemented

        return MultiplicativeConfiguratorChain(self, other)


class ConfigurationSampler(BaseConfigurationSampler):
    """
    Configuration sampler with a default implementation for __init__, contains_subset_of, contains_superset_of.
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

        # print(
        #     f"{type(self).__qualname__}.contains_subset_of", configuration, exclude
        # )

        if exclude is None:
            exclude = set()

        values = {
            k: v
            for k, v in self.configurator.parameter_values.items()
            if k not in exclude
        }

        exclude = exclude.union(self.configurator.parameter_values.keys())

        conf_parameters = configuration.get("parameters", {})

        # Check if all configured parameters are contained
        if any(
            k not in conf_parameters or conf_parameters[k] not in v
            for k, v in values.items()
        ):
            return False

        parent_params = {k: v for k, v in conf_parameters.items() if k not in values}

        # Let parents check the rest of the configuration
        return self.parent.contains_subset_of(
            dict(configuration, parameters=parent_params), exclude=exclude
        )

    def contains_superset_of(self, configuration: Mapping) -> bool:
        """
        Return True if there exists a sample that is a superset of `configuration`.

        - `configuration` does not match if it contains additional keys not produced by the sampler (or its parents).
        - `configuration` matches  if it lacks keys produced by the sampler.
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


class Configurator(BaseConfigurator):
    """Configurator with default implementation for build_sampler."""

    _Sampler: Type[ConfigurationSampler]

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        return self._Sampler(self, parent)


class MultiplicativeConfiguratorChain(Configurator):
    """
    Multiplicative configurator chain.
    The result is the cross-product of all contained configurators.
    """

    def __init__(self, *configurators: BaseConfigurator) -> None:
        self.configurators = configurators

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        for c in self.configurators:
            parent = c.build_sampler(parent)

        return parent

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        parameter_values = {}
        for c in self.configurators:
            replace_parameter_values(parameter_values, c.parameter_values)
        return parameter_values

    def __mul__(self, other) -> "BaseConfigurator":
        if not isinstance(other, BaseConfigurator):
            return NotImplemented

        return MultiplicativeConfiguratorChain(*self.configurators, other)


class AdditiveConfiguratorChain(Configurator):
    """
    Additive configurator chain.
    The result is the concatenation of all contained configurators.
    """

    def __init__(self, *configurators: BaseConfigurator) -> None:
        self.configurators = configurators

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        return self._Sampler(  # pylint: disable=no-member
            tuple(c.build_sampler(parent) for c in self.configurators)
        )

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        parameter_values = {}
        for c in self.configurators:
            extend_parameter_values(parameter_values, c.parameter_values)
        return parameter_values

    def __add__(self, other) -> "BaseConfigurator":
        if not isinstance(other, BaseConfigurator):
            return NotImplemented

        return AdditiveConfiguratorChain(*self.configurators, other)

    class _Sampler(BaseConfigurationSampler):
        def __init__(self, samplers: Iterable[BaseConfigurationSampler]) -> None:
            self.samplers = samplers

        def sample(self, exclude=None) -> Iterator[Mapping]:
            for s in self.samplers:
                yield from s.sample(exclude)  # pylint: disable=protected-access

        def contains_subset_of(
            self, configuration: Mapping, exclude: Optional[Set] = None
        ) -> bool:
            """Return True if there exists a sample that is a subset of `configuration`, i.e. if there is one in a child."""

            # print(
            #     f"{type(self).__qualname__}.contains_subset_of", configuration, exclude
            # )

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
    def __init__(self, values: Optional[Mapping[str, Any]] = None, **kwargs):
        if values is None:
            self.values = kwargs
        else:
            self.values = {**values, **kwargs}

    def __repr__(self):
        return f"<Const {self.values}>"

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {k: (v,) for k, v in self.values.items()}

    class _Sampler(ConfigurationSampler):
        configurator: "Const"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            if exclude is None:
                exclude = set()

            values = {
                k: v for k, v in self.configurator.values.items() if k not in exclude
            }

            exclude = exclude.union(self.configurator.values.keys())

            for parent_configuration in self.parent.sample(exclude=exclude):
                # print(self.configurator, ":", values)
                yield merge_dicts(parent_configuration, parameters=values)


def all_subsets(configuration: Mapping):
    """Generate all parameter subsets of a configuration to test Sampler.contains_superset_of."""
    parameters = configuration.get("parameters", {})

    key_sets = itertools.chain.from_iterable(
        itertools.combinations(parameters.keys(), l) for l in range(len(parameters) + 1)
    )

    return [
        dict(configuration, parameters={k: parameters[k] for k in keys})
        for keys in key_sets
    ]


configurator = Const({"a": 1}) * (Const({"b": 2}) + Const({"b": 3}))
sampler = configurator.build_sampler()

for conf in sampler:
    print(conf)
    assert sampler.contains_subset_of(conf), f"No subset of {conf} in sampler"

    superset = merge_dicts(conf, parameters={"__foo": "bar"})
    assert sampler.contains_subset_of(superset), f"No subset of {superset} in sampler"

    for subset in all_subsets(conf):
        assert sampler.contains_superset_of(
            subset
        ), f"No superset of {subset} in sampler"

# A non-empty sampler does not contain a subset of {}
assert not sampler.contains_subset_of({}), "Subset of {} in sampler"
# Every sampler contains a superset of {}
assert sampler.contains_superset_of({}), "No subset of {} in sampler"
