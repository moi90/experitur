from experitur.core.parameters import (
    ParameterGenerator,
    ParameterGeneratorIter,
    check_parameter_generators,
    Multi,
)
from experitur.helpers.merge_dicts import merge_dicts
from typing import Mapping, Any


class _ConditionsSamplerIter(ParameterGeneratorIter):
    def __iter__(self):
        for parent_configuration in self.parent:
            for value, sub_gen in self.parameter_generator.sub_generators.items():
                sub_gen: ParameterGenerator
                params = merge_dicts(
                    parent_configuration,
                    parameters={self.parameter_generator.name: value},
                )
                for sub_params in sub_gen.generate(self.experiment):
                    yield merge_dicts(params, sub_params)


def _check_sub_generators(sub_generators) -> Mapping[Any, ParameterGenerator]:
    result = {}

    for condition, sub_gen in sub_generators.items():
        sub_gen_list = check_parameter_generators(sub_gen)
        if not sub_gen_list:
            raise ValueError(
                f"Empty sub_generator for condition {condition!r}: {sub_gen}"
            )
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

            from experitur import Experiment, TrialParameters
            from experitur.parameters import Conditions, Grid

            @Conditions("a", {
                1: Grid({"b": [1,2]}),
                2: Grid({"b": [3,4]}),
            })
            @Experiment()
            def example(parameters: TrialParameters):
                print(parameters["a"], parameters["b"])

        This example will produce "1 1", "1 2", "2 3", "2 4".
    """

    _iterator = _ConditionsSamplerIter
    _str_attr = ["n_iter"]

    def __init__(self, name, sub_generators: Mapping):
        self.name = name
        self.sub_generators = _check_sub_generators(sub_generators)
        self._varying_parameters = None
        self._invariant_parameters = None

    @property
    def varying_parameters(self):
        if self._varying_parameters is not None:
            return self._varying_parameters

        varying_parameters = {self.name: list(self.sub_generators.keys())}

        for sub in self.sub_generators.values():
            for k, v in sub.varying_parameters.items():
                varying_parameters.setdefault(k, list()).append(v)

        self._varying_parameters = varying_parameters

        return varying_parameters

    @property
    def invariant_parameters(self):
        if self._invariant_parameters is not None:
            return self._invariant_parameters

        invariant_parameters = {}

        for sub in self.sub_generators.values():
            for k, v in sub.invariant_parameters.items():
                if k not in self.varying_parameters:
                    invariant_parameters.setdefault(k, list()).append(v)

        self._invariant_parameters = invariant_parameters

        return invariant_parameters
