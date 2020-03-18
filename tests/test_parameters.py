import pytest

from experitur import Experiment, TrialParameters
from experitur.core.context import Context
from experitur.parameters import Random, SKOpt


def test_Random(tmp_path):
    with Context(str(tmp_path)) as ctx:
        sampler = Random({"a": [1, 2], "b": [3, 4], "c": [0]}, 4)

        # Test __str__
        str(sampler)

        @Experiment(parameters=sampler)
        def exp(trial):
            pass

        sample_iter = sampler.generate(exp)
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


def test_RandomRepeat(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config) as ctx:
        parameters = Random({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}, 4)

        @Experiment(parameters=parameters)
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
        parameters.n_iter = 8
        ctx.run()

        # New trials should have been introduced
        # (It might be less than 8 because the same values might have been drawn againg.)
        c_trials = ctx.store.match(experiment=exp)
        assert 4 <= len(c_trials) <= 8


def test_SKOpt(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config) as ctx:
        parameters = SKOpt({"x": (-10.0, 10.0, "uniform")}, "x", 4)

        @Experiment(parameters=parameters)
        def exp(trial: TrialParameters):
            return dict(trial)

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
        parameters.n_iter = 8
        ctx.run()

        # New trials should have been introduced
        # (It might be less than 8 because the same values might have been drawn againg.)
        c_trials = ctx.store.match(experiment=exp)
        assert 4 <= len(c_trials) <= 8