from experitur.testing.configurators import (
    assert_sampler_contains_subset_of_all_samples,
    assert_sampler_contains_superset_of_all_samples,
)
import pytest

from experitur import Context, Experiment

try:
    import numpy as np
    from scipy.stats import distributions

    from experitur.configurators.random import _DistWrapper, Random
except ImportError as exc:
    pytestmark = pytest.mark.skip(str(exc))


def test_DistWrapper():
    dist = _DistWrapper(distributions.uniform(0, 1))
    for x in np.linspace(0, 1, 10):
        assert x in dist

    assert -0.1 not in dist
    assert 1.1 not in dist

    dist = _DistWrapper(distributions.norm())
    for x in np.linspace(-10, 10, 10):
        assert x in dist


def test_Random(tmp_path):
    with Context(str(tmp_path)) as ctx:
        configurator = Random({"a": [1, 2], "b": [3, 4], "c": [0]}, 4)

        # Test __str__
        str(configurator)

        @Experiment(configurator=configurator)
        def exp(trial):
            pass

        with ctx.set_current_experiment(exp):
            sampler = configurator.build_sampler()

            assert_sampler_contains_subset_of_all_samples(sampler, {"d": 5})
            assert_sampler_contains_superset_of_all_samples(sampler)

            samples = set(tuple(sorted(d["parameters"].items())) for d in sampler)

            # Assert exististence of all grid  cells
            assert samples == {
                (("a", 2), ("b", 4), ("c", 0)),
                (("a", 1), ("b", 4), ("c", 0)),
                (("a", 2), ("b", 3), ("c", 0)),
                (("a", 1), ("b", 3), ("c", 0)),
            }

            # Assert correct behavior of independent_parameters
            assert configurator.parameter_values == {
                "a": (1, 2),
                "b": (3, 4),
                "c": (0,),
            }


def test_RandomRepeat(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config, writable=True) as ctx:
        configurator = Random({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}, 4)

        @Experiment(configurator=configurator)
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
        configurator.n_iter = 8
        ctx.run()

        # New trials should have been introduced
        # (It might be less than 8 because the same values might have been drawn againg.)
        c_trials = ctx.store.match(experiment=exp)
        assert 4 <= len(c_trials) <= 8
