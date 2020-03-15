import pytest

from experitur.core.context import push_context, Context
from experitur.core.samplers import GridSampler


def test_merge(tmp_path):
    config = {"skip_existing": False}
    with push_context(Context(str(tmp_path), config)) as ctx:

        sampler = GridSampler({"a": [1, 2]})

        @ctx.experiment("a", sampler=sampler, meta={"a": "foo"})
        def a(trial):
            pass

        b = ctx.experiment("b", parent=a)

        # Ensure that meta is *copied* to child experiments
        assert id(b.meta) != id(a.meta)
        assert b.meta == a.meta

        # Assert that samplers are concatenated in the right way
        c = ctx.experiment("c", sampler=GridSampler({"b": [1, 2]}), parent=a)
        ctx.run()

        # Parameters in a and b should be the same
        a_params = set(
            tuple(t.data["parameters"].items())
            for t in ctx.store.match(experiment=a).values()
        )
        b_params = set(
            tuple(t.data["parameters"].items())
            for t in ctx.store.match(experiment=b).values()
        )

        assert a_params == b_params

        print(ctx.store.match())

        c_trials = ctx.store.match(experiment=c)

        assert len(c_trials) == 4

        parameter_configurations = set(
            tuple(t.data["parameters"].items()) for t in c_trials.values()
        )

        # Assert exististence of all grid  cells
        assert parameter_configurations == {
            (("a", 2), ("b", 1)),
            (("a", 1), ("b", 1)),
            (("a", 2), ("b", 2)),
            (("a", 1), ("b", 2)),
        }


def test_failing_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with push_context(Context(str(tmp_path), config)) as ctx:

        @ctx.experiment()
        def experiment(trial):
            raise Exception("Some error")

        with pytest.raises(Exception):
            ctx.run()

        trial_id, trial = ctx.store.popitem()

        print(trial.data)

        assert trial.data["success"] == False
