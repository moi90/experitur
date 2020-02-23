from experitur.samplers import (
    GridSampler,
    RandomSampler,
    parameter_product,
    MultiSampler,
)
from experitur import context
import pytest


def test_empty_parameter_product():
    assert list(parameter_product({})) == [{}]


@pytest.mark.parametrize("shuffle", [True, False])
def test_GridSampler(tmp_path, shuffle):
    with context.push_context(context.Context(str(tmp_path))) as ctx:
        sampler = GridSampler({"a": [1, 2], "b": [3, 4], "c": [0]}, shuffle=shuffle)

        sample_iter = sampler.sample(ctx.store)
        samples = set(tuple(d.items()) for d in sample_iter)

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 2), ("b", 4), ("c", 0)),
            (("a", 1), ("b", 4), ("c", 0)),
            (("a", 2), ("b", 3), ("c", 0)),
            (("a", 1), ("b", 3), ("c", 0)),
        }

        # Assert correct behavior of "parameters"
        assert sampler.parameters == {"a": [1, 2], "b": [3, 4]}

        assert sampler.invariant_parameters == {"c": [0]}


def test_RandomSampler(tmp_path):
    with context.push_context(context.Context(str(tmp_path))) as ctx:
        sampler = RandomSampler({"a": [1, 2], "b": [3, 4], "c": [0]}, 4)

        sample_iter = sampler.sample(ctx.store)
        samples = set(tuple(sorted(d.items())) for d in sample_iter)

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 2), ("b", 4), ("c", 0)),
            (("a", 1), ("b", 4), ("c", 0)),
            (("a", 2), ("b", 3), ("c", 0)),
            (("a", 1), ("b", 3), ("c", 0)),
        }

        # Assert correct behavior of "parameters"
        assert sampler.parameters == {"a": [1, 2], "b": [3, 4]}

        assert sampler.invariant_parameters == {"c": [0]}


def test_MultiSampler(tmp_path):
    with context.push_context(context.Context(str(tmp_path))) as ctx:
        sampler = MultiSampler(
            [
                GridSampler({"a": [1, 2], "b": [3, 4], "c": [10, 11]}),
                GridSampler({"a": [4, 5], "c": [0]}),
            ]
        )

        sample_iter = sampler.sample(ctx.store)
        samples = set(tuple(sorted(d.items())) for d in sample_iter)

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 4), ("b", 4), ("c", 0)),
            (("a", 5), ("b", 4), ("c", 0)),
            (("a", 4), ("b", 3), ("c", 0)),
            (("a", 5), ("b", 3), ("c", 0)),
        }

        # Assert correct behavior of "parameters"
        assert sampler.parameters == {"a": [4, 5], "b": [3, 4]}
