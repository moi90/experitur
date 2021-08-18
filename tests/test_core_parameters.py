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

        sample_iter = sampler.generate()
        samples = set(
            tuple(configuration["parameters"].items()) for configuration in sample_iter
        )

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 1), ("b", 2), ("c", 3)),
        }

        # Assert correct behavior of "independent_parameters"
        assert sampler.independent_parameters == {"a": [1], "b": [2], "c": [3]}


@pytest.mark.parametrize("shuffle", [True, False])
def test_Grid(tmp_path, shuffle):
    with Context(str(tmp_path)):
        sampler = Grid({"a": [1, 2], "b": [3, 4], "c": [0]}, shuffle=shuffle)

        # Test __str__
        str(sampler)

        @Experiment(parameters=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.generate()
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

        # Assert correct behavior of "independent_parameters"
        assert sampler.independent_parameters == {"a": [1, 2], "b": [3, 4], "c": [0]}


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

        sample_iter = sampler.generate()
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
        assert sampler.independent_parameters == {"a": [4, 5], "b": [3, 4], "c": [0]}


def test_Sequential(tmp_path):
    with Context(str(tmp_path)):
        generator = Grid({"a": [1, 2], "b": [3, 4], "c": [10, 11]}) + Grid(
            {"a": [4, 5], "c": [0]}
        )

        # Test __str__
        str(generator)

        @Experiment(parameters=generator)
        def exp(trial):
            pass

        sample_iter = generator.generate()
        samples = sorted(
            tuple(sorted(configuration["parameters"].items()))
            for configuration in sample_iter
        )

        assert len(samples) == 10

        # Assert exististence of all grid cells
        assert samples == [
            # First Grid
            (("a", 1), ("b", 3), ("c", 10)),
            (("a", 1), ("b", 3), ("c", 11)),
            (("a", 1), ("b", 4), ("c", 10)),
            (("a", 1), ("b", 4), ("c", 11)),
            (("a", 2), ("b", 3), ("c", 10)),
            (("a", 2), ("b", 3), ("c", 11)),
            (("a", 2), ("b", 4), ("c", 10)),
            (("a", 2), ("b", 4), ("c", 11)),
            # Second Grid
            (("a", 4), ("c", 0)),
            (("a", 5), ("c", 0)),
        ]

        # Assert correct behavior of "independent_parameters"
        assert generator.independent_parameters == {
            "a": [1, 2, 4, 5],
            "b": [3, 4],
            "c": [10, 11, 0],
        }
