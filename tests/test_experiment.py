import pytest

from experitur import context


def test_merge():
    with context.push_context() as ctx:
        @ctx.experiment("a", parameter_grid={"a": [1, 2]})
        def a(trial):
            pass

        b = ctx.experiment("b", parent=a)

        assert b.parameter_grid == a.parameter_grid

        # Check that the parameters specified here override the parent parameters
        c = ctx.experiment("c", parameter_grid={
                           "a": [3, 4], "b": [0]}, parent=a)
        assert c.parameter_grid == {"a": [3, 4], "b": [0]}


def test_failing_experiment(tmp_path):
    config = {
        "catch_exceptions": False
    }
    with context.push_context(context.Context(str(tmp_path), config)) as ctx:
        @ctx.experiment()
        def experiment(trial):
            raise Exception("Some error")

        with pytest.raises(Exception):
            ctx.run()

        trial_id, trial = ctx.store.popitem()

        print(trial.data)

        assert trial.data["success"] == False
