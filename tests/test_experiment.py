import pytest

from experitur import context


def test_merge():
    with context.push_context() as ctx:
        @ctx.experiment("a", parameter_grid={"a": [1, 2]})
        def a(trial):
            pass

        b = ctx.experiment("b", parent=a)

        assert b.parameter_grid == a.parameter_grid


def test_failing_experiment():
    with context.push_context() as ctx:
        @ctx.experiment()
        def experiment(trial):
            raise Exception("Some error")

        ctx.run()

        trial_id, trial = ctx.store.popitem()

        print(trial.data)

        assert trial.data["success"] == False
