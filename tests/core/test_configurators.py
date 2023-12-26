from typing import Union

import pytest
from experitur.core.configurators import (
    Const,
    Grid,
    RandomGrid,
    Clear,
    parameter_product,
)
from experitur.testing.configurators import (
    assert_sampler_contains_subset_of_all_samples,
    assert_sampler_contains_superset_of_all_samples,
    sampler_parameter_values,
)
from experitur.util import unset


def test_empty_parameter_product():
    assert list(parameter_product({})) == [{}]


def test_Const():
    configurator = Const({"a": 1, "b": 2}, c=3, d=unset)

    # Test __str__
    str(configurator)

    sampler = configurator.build_sampler()

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == {
        "a": (1,),
        "b": (2,),
        "c": (3,),
        "d": (unset,),
    }

    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = set(
        tuple(configuration["parameters"].items()) for configuration in sampler
    )

    # Assert exististence of all grid  cells
    assert samples == {
        (("a", 1), ("b", 2), ("c", 3), ("d", unset)),
    }


@pytest.mark.parametrize("cls", [Grid, RandomGrid])
def test_Grid(cls):
    configurator: Union[Grid, RandomGrid] = cls(
        {"a": [1, 2], "b": [3, 4], "c": [0], "d": [0, unset]}
    )

    # Test __str__
    str(configurator)

    sampler = configurator.build_sampler()

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == {
        "a": (1, 2),
        "b": (3, 4),
        "c": (0,),
        "d": (0, unset),
    }

    # Test contains_subset_of and contains_superset_of
    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)

    samples = set(
        tuple(configuration["parameters"].items()) for configuration in sampler
    )

    # Assert exististence of all grid  cells
    assert samples == {
        (("a", 2), ("b", 4), ("c", 0), ("d", 0)),
        (("a", 1), ("b", 4), ("c", 0), ("d", 0)),
        (("a", 2), ("b", 3), ("c", 0), ("d", 0)),
        (("a", 1), ("b", 3), ("c", 0), ("d", 0)),
        (("a", 2), ("b", 4), ("c", 0), ("d", unset)),
        (("a", 1), ("b", 4), ("c", 0), ("d", unset)),
        (("a", 2), ("b", 3), ("c", 0), ("d", unset)),
        (("a", 1), ("b", 3), ("c", 0), ("d", unset)),
    }


def test_MultiplicativeConfiguratorChain():
    configurator = Grid({"a": [1, 2], "b": [3, 4], "c": [10, 11]}) * Grid(
        {"a": [4, 5], "c": [0]}
    )

    # Test __str__
    str(configurator)

    sampler = configurator.build_sampler()

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == {"a": (4, 5), "b": (3, 4), "c": (0,)}

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

    sampler = configurator.build_sampler()

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == {
        "a": (1, 2, 4, 5),
        "b": (3, 4, unset),
        "c": (10, 11, 0),
    }

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

    sampler = configurator.build_sampler()

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == {
        "a": {1, 2, 3},
        "b": {unset, 1},
    }


def test_Unset():
    # FIXME: Unset is currently inherently broken.
    # The problem is that Unset("b") does not have access to the outer parameter b
    # during sampler.parameter_values
    # Solution: Shift parameter_values to sampler, so it has access to the parent's parameter_values.
    # This way, all the parameter_values building logic is moved from a global to the specific sampler.

    configurator = Const(a=1, b=2, c=3) * (Const() + (Const(a=2) * Clear("b")))
    # configurator = Const(a=1, b=2) * Unset("b")

    # Test __str__
    str(configurator)

    sampler = configurator.build_sampler()

    parameter_values_expected = {
        "a": (1, 2),
        "b": (2, unset),
        "c": (3,),
    }

    # Assert correct behavior of "parameter_values"
    assert sampler.parameter_values == parameter_values_expected

    assert sampler_parameter_values(sampler) == parameter_values_expected

    # Test contains_subset_of and contains_superset_of
    assert_sampler_contains_subset_of_all_samples(sampler, include_parameters={"d": 4})
    assert_sampler_contains_superset_of_all_samples(sampler)
