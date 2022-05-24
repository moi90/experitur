import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.configurators import Grid


def test_trial_collection(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(configurator=Grid({"a": [1, 2], "b": [3, 4], "c": [5]}))
        def a(trial):
            pass

    # Run experiments
    ctx.run()

    trials = ctx.trials.match(experiment=a)

    assert trials.varying_parameters == {"a": {1, 2}, "b": {3, 4}}
    assert trials.invariant_parameters == {"c": 5}


def test_groupby(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment(configurator=Grid({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
        def a(trial):
            return dict(trial)

        @Experiment(parent=a)
        def b(trial):
            return dict(trial)

    ctx.run()

    trials = ctx.trials.match()

    groups = trials.groupby(parameters="a")

    assert len(groups) == 2

    for group_key, group in groups:
        assert set(group_key.keys()) == {"a"}
        assert group_key["a"] in (1, 2)

        assert "a" not in group.varying_parameters

    result = groups.filter(lambda trial: trial["b"] == 3).best_n(10, maximize="c")

    assert len(result) == 2

    trials = result.coalesce()

    assert set(trials.varying_parameters.keys()) == {"a", "c"}


def test_pareto_optimal(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment(configurator=Grid({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
        def a(trial):
            return dict(trial)

        @Experiment(parent=a)
        def b(trial):
            return dict(trial)

    ctx.run()

    trials = ctx.trials.match()

    trials.pareto_optimal(minimize=["a", "b"], maximize="c")
