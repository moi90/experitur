import pytest

from experitur.core.experiment import Experiment, format_trial_parameters
from experitur.core.context import Context
from experitur.core.parameters import Grid, ParameterGenerator


def test_trial_collection(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config) as ctx:

        @Experiment(parameters=Grid({"a": [1, 2], "b": [3, 4], "c": [5]}))
        def a(trial):
            pass

    # Run experiments
    ctx.run()

    trials = ctx.get_trials(experiment=a)

    assert trials.varying_parameters == {"a", "b"}
