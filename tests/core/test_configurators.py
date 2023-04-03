from typing import Union

import pytest

from experitur.core.configurators import (
    Const,
    Grid,
    RandomGrid,
    parameter_product,
)
from experitur.testing.configurators import (
    assert_sampler_contains_subset_of_all_samples,
    assert_sampler_contains_superset_of_all_samples,
)
from experitur.util import unset


def test_empty_parameter_product():
    assert list(parameter_product({})) == [{}]


def test_Const():
    configurator = Const({"a": 1, "b": 2}, c=3)

    # Test __str__
    str(configurator)

    # Assert correct behavior of "parameter_values"
    assert configurator.parameter_values == {"a": (1,), "b": (2,), "c": (3,)}

    sampler = configurator.build_sampler()

    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = set(
        tuple(configuration["parameters"].items()) for configuration in sampler
    )

    # Assert exististence of all grid  cells
    assert samples == {
        (("a", 1), ("b", 2), ("c", 3)),
    }


@pytest.mark.parametrize("cls", [Grid, RandomGrid])
def test_Grid(cls):
    configurator: Union[Grid, RandomGrid] = cls({"a": [1, 2], "b": [3, 4], "c": [0]})

    # Test __str__
    str(configurator)

    # Assert correct behavior of "parameter_values"
    assert configurator.parameter_values == {"a": (1, 2), "b": (3, 4), "c": (0,)}

    sampler = configurator.build_sampler()

    # Test contains_subset_of and contains_superset_of
    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = set(
        tuple(configuration["parameters"].items()) for configuration in sampler
    )

    # Assert exististence of all grid  cells
    assert samples == {
        (("a", 2), ("b", 4), ("c", 0)),
        (("a", 1), ("b", 4), ("c", 0)),
        (("a", 2), ("b", 3), ("c", 0)),
        (("a", 1), ("b", 3), ("c", 0)),
    }


def test_MultiplicativeConfiguratorChain():
    configurator = Grid({"a": [1, 2], "b": [3, 4], "c": [10, 11]}) * Grid(
        {"a": [4, 5], "c": [0]}
    )

    # Test __str__
    str(configurator)

    # Assert correct behavior of "parameter_values"
    assert configurator.parameter_values == {"a": (4, 5), "b": (3, 4), "c": (0,)}

    sampler = configurator.build_sampler()

    # Test contains_subset_of and contains_superset_of
    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = sorted(
        tuple(sorted(configuration["parameters"].items())) for configuration in sampler
    )

    assert len(samples) == 4

    # Assert exististence of all grid cells
    assert samples == [
        (("a", 4), ("b", 3), ("c", 0)),
        (("a", 4), ("b", 4), ("c", 0)),
        (("a", 5), ("b", 3), ("c", 0)),
        (("a", 5), ("b", 4), ("c", 0)),
    ]


def test_AdditiveConfiguratorChain():
    configurator = Grid({"a": [1, 2], "b": [3, 4], "c": [10, 11]}) + Grid(
        {"a": [4, 5], "c": [0]}
    )

    # Test __str__
    str(configurator)

    # Assert correct behavior of "parameter_values"
    assert configurator.parameter_values == {
        "a": (1, 2, 4, 5),
        "b": (3, 4, unset),
        "c": (10, 11, 0),
    }

    sampler = configurator.build_sampler()

    # Test contains_subset_of and contains_superset_of
    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = sorted(
        tuple(sorted(configuration["parameters"].items())) for configuration in sampler
    )

    assert len(samples) == 10

    # Assert exististence of all samples
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


def test_AdditiveConst():
    configurator = Const(a=1) * (Const() + Const(a=2, b=1) + Const(a=3, b=1))

    # Assert correct behavior of "parameter_values"
    assert configurator.parameter_values == {
        "a": (1, 2, 3),
        "b": (unset, 1),
    }
