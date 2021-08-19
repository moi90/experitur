from typing import Any, Container, List, Mapping, Optional, Union

from experitur.core.configurators import (
    AdditiveConfiguratorChain,
    BaseConfigurationSampler,
    BaseConfigurator,
    Configurator,
    Const,
    MultiplicativeConfiguratorChain,
    _RootSampler,
    extend_parameter_values,
    is_invariant,
    validate_configurators,
)
from experitur.helpers.merge_dicts import merge_dicts


def _validate_conditions(conditions, name) -> Mapping[Any, BaseConfigurator]:
    result = {}

    for condition, c in conditions.items():
        c_list = validate_configurators(c)

        result[condition] = MultiplicativeConfiguratorChain(
            # name comes last to overwrite configured values
            *c_list,
            Const({name: condition})
        )

    return result


class _Conditions(Configurator):
    _Sampler = AdditiveConfiguratorChain._Sampler

    def __init__(self, conditions: Mapping[Any, BaseConfigurator], name: str) -> None:
        self.conditions = conditions
        self.name = name

    def merge(self, other_conditions):
        conditions = dict(self.conditions)

        other_sub_generators = _validate_conditions(other_conditions, self.name)

        for k, v2 in other_sub_generators.items():
            try:
                v1 = conditions[k]
            except IndexError:
                continue

            conditions[k] = v1 * v2

        return _Conditions(conditions, self.name)

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        parameter_values = {}
        for c in self.conditions.values():
            extend_parameter_values(parameter_values, c.parameter_values)
        return parameter_values

    def build_sampler(
        self, parent: Optional["BaseConfigurationSampler"] = None
    ) -> BaseConfigurationSampler:
        if parent is None:
            parent = _RootSampler()

        return self._Sampler(  # pylint: disable=no-member
            tuple(c.build_sampler(parent) for c in self.conditions.values())
        )


class Conditions(_Conditions):
    """
    Different conditions, each with their own parameter configurations.

    Parameters:
        name (str): Parameter that stores the respective condition.
        conditions (Mapping): Mapping of conditions to sub-generators.
        active (list of str, optional): List of active conditions.


    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.configurators import Conditions, Grid

            @Conditions("a", {
                1: Grid({"b": [1,2]}),
                2: Grid({"b": [3,4]}),
            })
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        This example will produce "1 1", "1 2", "2 3", "2 4".
    """

    def __init__(self, name, conditions: Mapping, active: Optional[List[str]] = None):
        if isinstance(active, str):
            active = [active]

        conditions = _validate_conditions(conditions, name)

        if active is not None:
            conditions = {k: conditions[k] for k in active}

        super().__init__(conditions, name)
