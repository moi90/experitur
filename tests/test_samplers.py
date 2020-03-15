import pytest

from experitur.core.context import Context, push_context
from experitur.core.samplers import (
    GridSampler,
    MultiSampler,
    RandomSampler,
    parameter_product,
)


def test_empty_parameter_product():
    assert list(parameter_product({})) == [{}]


@pytest.mark.parametrize("shuffle", [True, False])
def test_GridSampler(tmp_path, shuffle):
    with push_context(Context(str(tmp_path))) as ctx:
        sampler = GridSampler({"a": [1, 2], "b": [3, 4], "c": [0]}, shuffle=shuffle)

        # Test __str__
        str(sampler)

        @ctx.experiment(sampler=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.sample(exp)
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


def test_RandomSampler(tmp_path):
    with push_context(Context(str(tmp_path))) as ctx:
        sampler = RandomSampler({"a": [1, 2], "b": [3, 4], "c": [0]}, 4)

        # Test __str__
        str(sampler)

        @ctx.experiment(sampler=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.sample(exp)
        samples = set(tuple(sorted(d["parameters"].items())) for d in sample_iter)

        # Assert exististence of all grid  cells
        assert samples == {
            (("a", 2), ("b", 4), ("c", 0)),
            (("a", 1), ("b", 4), ("c", 0)),
            (("a", 2), ("b", 3), ("c", 0)),
            (("a", 1), ("b", 3), ("c", 0)),
        }

        # Assert correct behavior of `parameters` and `invariant_parameters`
        assert sampler.varying_parameters == {"a": [1, 2], "b": [3, 4]}
        assert sampler.invariant_parameters == {"c": [0]}


def test_RandomSamplerRepeat(tmp_path):
    config = {"skip_existing": True}
    with push_context(Context(str(tmp_path), config)) as ctx:
        sampler = RandomSampler({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}, 4)

        @ctx.experiment(sampler=sampler)
        def exp(trial):
            pass

        # Execute experiment a first time
        ctx.run()

        c_trials = ctx.store.match(experiment=exp)

        assert len(c_trials) == 4

        # Execute experiment a second time
        ctx.run()

        # No new trials should have been introduced
        c_trials = ctx.store.match(experiment=exp)
        assert len(c_trials) == 4

        # Increase number of trials and rerun a third time
        sampler.n_iter = 8
        ctx.run()

        # New trials should have been introduced
        # (It might be less than 8 because the same values might have been drawn againg.)
        c_trials = ctx.store.match(experiment=exp)
        assert 4 <= len(c_trials) <= 8


def test_MultiGrid(tmp_path):
    with push_context(Context(str(tmp_path))) as ctx:
        sampler = MultiSampler(
            [
                GridSampler({"a": [1, 2], "b": [3, 4], "c": [10, 11]}),
                GridSampler({"a": [4, 5], "c": [0]}),
            ]
        )

        # Test __str__
        str(sampler)

        @ctx.experiment(sampler=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.sample(exp)
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
