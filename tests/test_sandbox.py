import inspect

import pytest

from experitur.core.context import Context
from experitur.core.trial import Trial
from experitur.sandbox import WorkerPool

example_py = inspect.cleandoc(
    r"""
    from experitur import Experiment

    @Experiment()
    def baseline(parameters):
        return {"foo": "bar"}
    """
)


@pytest.mark.xfail
def test_sandbox(tmp_path):
    dox_fn = str(tmp_path / "example.py")
    with open(dox_fn, "w") as f:
        f.write(example_py)

    pool = WorkerPool(dox_fn)

    ctx = Context(str(tmp_path), writable=True)
    trial = ctx.trials.create(
        {"experiment": {"name": "test_experiment", "varying_parameters": []}}
    )

    pool.run_trial(trial)
