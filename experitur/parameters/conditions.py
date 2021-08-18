from typing import Any, List, Mapping, Optional, Union

from experitur.core.parameters import (
    DynamicValues,
    Multi,
    ParameterGenerator,
    ParameterGeneratorIter,
    check_parameter_generators,
    count_values,
)
from experitur.helpers.merge_dicts import merge_dicts


class _ConditionsSamplerIter(ParameterGeneratorIter):
    def __iter__(self):
        for parent_configuration in self.parent:
            for value, sub_gen in self.parameter_generator.sub_generators.items():
                sub_gen: ParameterGenerator
                for sub_params in sub_gen.generate():
                    yield merge_dicts(
                        parent_configuration,
                        sub_params,
                        parameters={self.parameter_generator.name: value},
                    )


def _check_sub_generators(sub_generators) -> Mapping[Any, ParameterGenerator]:
    result = {}

    for condition, sub_gen in sub_generators.items():
        sub_gen_list = check_parameter_generators(sub_gen)

        if len(sub_gen_list) == 1:
            result[condition] = sub_gen_list[0]
        else:
            result[condition] = Multi(sub_gen_list)

    return result


class Conditions(ParameterGenerator):
    """
    Different conditions, each with their own parameter configurations.

    Parameters:
        name (str): Parameter that stores the respective condition.
        sub_generators (Mapping): Mapping of conditions to sub-generators.


    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.parameters import Conditions, Grid

            @Conditions("a", {
                1: Grid({"b": [1,2]}),
                2: Grid({"b": [3,4]}),
            })
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

        This example will produce "1 1", "1 2", "2 3", "2 4".
    """

    _iterator = _ConditionsSamplerIter
    _str_attr = ["n_iter"]

    def __init__(
        self, name, sub_generators: Mapping, active: Optional[List[str]] = None
    ):
        self.name = name

        if isinstance(active, str):
            active = [active]

        self.active = active

        sub_generators = _check_sub_generators(sub_generators)
        if active is not None:
            sub_generators = {k: sub_generators[k] for k in active}

        self.sub_generators = sub_generators

        self._varying_parameters = None
        self._invariant_parameters = None

    def merge(self, other_sub_generators):
        sub_generators = self.sub_generators.copy()

        for k, v2 in other_sub_generators.items():
            v1 = sub_generators.get(k)
            if v1 is None:
                sub_generators[k] = v2
                continue

            v1 = check_parameter_generators(v1)
            v2 = check_parameter_generators(v2)

            sub_generators[k] = Multi(v1 + v2)

        return Conditions(self.name, sub_generators, self.active)

    @property
    def independent_parameters(self) -> Mapping[str, Union[List, DynamicValues]]:
        independent_parameters = {self.name: list(self.sub_generators.keys())}

        include = set([self.name])
        for sub in self.sub_generators.values():
            for k, v in sub.independent_parameters.items():
                # Skip entries that are overwritten by the configuration name
                if k == self.name:
                    continue

                # Include only those that vary independently of the condition
                length = count_values(v)
                if length is None or length > 1:
                    include.add(k)

                try:
                    independent_parameters[k] = independent_parameters.get(k, []) + v
                except:
                    print(f"Error concatenating {sub}")
                    raise

        return {k: v for k, v in independent_parameters.items() if k in include}
