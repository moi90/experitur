import pytest

from experitur import Experiment, Trial
from experitur.core.context import Context
from experitur.parameters import Random, SKOpt


def test_SKOpt(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config, writable=True) as ctx:
        parameters = SKOpt({"x": (-10.0, 10.0, "uniform"), "y": (0, 10)}, "x", 4)

        @Experiment(parameters=parameters, minimize="x")
        def exp(trial: Trial):
            assert type(trial["x"]) is float
            assert type(trial["y"]) is int

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


def test_SKOptTimed(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config, writable=True) as ctx:
        parameters = SKOpt(
            {"x": (-10.0, 10.0, "uniform"), "y": (0, 10)}, "x", 4, acq_func="EIps"
        )

        @Experiment(parameters=parameters, minimize="x")
        def exp(trial: Trial):
            return dict(trial)

        # Execute experiment a first time
        ctx.run()

        c_trials = ctx.store.match(experiment=exp)
        assert len(c_trials) == 4
