from experitur.core.experiment import Experiment
from experitur.core.context import Context
from experitur.core.parameters import Grid


def test_trial_collection(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(parameters=Grid({"a": [1, 2], "b": [3, 4], "c": [5]}))
        def a(trial):
            pass

    # Run experiments
    ctx.run()

    trials = ctx.get_trials(experiment=a)

    assert trials.varying_parameters == {"a": [1, 2], "b": [3, 4]}
    assert trials.invariant_parameters == {"c": [5]}


def test_groupby(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment(parameters=Grid({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
        def a(trial):
            pass

        @Experiment(parent=a)
        def b(trial):
            pass

    ctx.run()

    trials = ctx.get_trials()

    for group_key, group in trials.groupby(parameters="a"):
        assert set(group_key.keys()) == set(["a"])
        assert group_key["a"] in (1, 2)

        assert "a" not in group.varying_parameters
