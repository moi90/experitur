import pytest

from experitur import context
from experitur.samplers import GridSampler


def test_merge():
    with context.push_context() as ctx:

        sampler = GridSampler({"a": [1, 2]})

        @ctx.experiment("a", sampler=sampler, meta={"a": "foo"})
        def a(trial):
            pass

        b = ctx.experiment("b", parent=a)

        assert b.sampler == a.sampler

        # Ensure that meta is *copied* to child experiments
        assert id(b.meta) != id(a.meta)
        assert b.meta == a.meta


def test_failing_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with context.push_context(context.Context(str(tmp_path), config)) as ctx:

        @ctx.experiment()
        def experiment(trial):
            raise Exception("Some error")

        with pytest.raises(Exception):
            ctx.run()

        trial_id, trial = ctx.store.popitem()

        print(trial.data)

        assert trial.data["success"] == False
