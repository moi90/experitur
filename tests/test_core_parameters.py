import pytest

from experitur import Experiment
from experitur.core.context import Context
from experitur.core.parameters import Const, Grid, Multi, parameter_product


def test_empty_parameter_product():
    assert list(parameter_product({})) == [{}]


def test_Const(tmp_path):
    with Context(str(tmp_path)):
        Const(c=3)

        sampler = Const({"a": 1, "b": 2}, c=3)

        # Test __str__
        str(sampler)

        @Experiment(parameters=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.generate(exp)
        samples = set(
            tuple(configuration["parameters"].items()) for configuration in sample_iter
        )

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 1), ("b", 2), ("c", 3)),
        }

        # Assert correct behavior of "parameters"
        assert sampler.varying_parameters == {}

        assert sampler.invariant_parameters == {"a": 1, "b": 2, "c": 3}


@pytest.mark.parametrize("shuffle", [True, False])
def test_Grid(tmp_path, shuffle):
    with Context(str(tmp_path)):
        sampler = Grid({"a": [1, 2], "b": [3, 4], "c": [0]}, shuffle=shuffle)

        # Test __str__
        str(sampler)

        @Experiment(parameters=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.generate(exp)
        samples = set(
            tuple(configuration["parameters"].items()) for configuration in sample_iter
        )

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 2), ("b", 4), ("c", 0)),
            (("a", 1), ("b", 4), ("c", 0)),
            (("a", 2), ("b", 3), ("c", 0)),
            (("a", 1), ("b", 3), ("c", 0)),
        }

        # Assert correct behavior of "parameters"
        assert sampler.varying_parameters == {"a": [1, 2], "b": [3, 4]}

        assert sampler.invariant_parameters == {"c": [0]}


def test_Multi(tmp_path):
    with Context(str(tmp_path)):
        sampler = Multi(
            [
                Grid({"a": [1, 2], "b": [3, 4], "c": [10, 11]}),
                Grid({"a": [4, 5], "c": [0]}),
            ]
        )

        # Test __str__
        str(sampler)

        @Experiment(parameters=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.generate(exp)
        samples = sorted(
            tuple(sorted(configuration["parameters"].items()))
            for configuration in sample_iter
        )

        assert len(samples) == 4

        # Assert exististence of all grid cells
        assert samples == [
            (("a", 4), ("b", 3), ("c", 0)),
            (("a", 4), ("b", 4), ("c", 0)),
            (("a", 5), ("b", 3), ("c", 0)),
            (("a", 5), ("b", 4), ("c", 0)),
        ]

        # Assert correct behavior of "parameters"
        assert sampler.varying_parameters == {"a": [4, 5], "b": [3, 4]}
