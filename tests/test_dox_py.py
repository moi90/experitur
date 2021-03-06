import inspect
import os

import pytest

from experitur.core.context import Context
from experitur.dox import DOXError, load_dox


@pytest.fixture(name="dox_py_fn")
def fixture_dox_py_fn(tmp_path):
    fn = str(tmp_path / "dox.py")
    with open(fn, "w") as f:
        f.write(
            inspect.cleandoc(
                """
                from experitur import Experiment

                @Experiment(
                    parameters={
                        "a1": [1],
                        "a2": [2],
                        "b": [1, 2],
                        "a": ["{a_{b}}"],
                    })
                def baseline(trial):
                    return trial.parameters

                # This experiment shouldn't be executed, because this combination of callable and parameters was already executed.
                Experiment(
                    "second_experiment",
                    parent=baseline
                )
                """
            )
        )

    return fn


def test_dox_py(dox_py_fn):
    wdir = os.path.splitext(dox_py_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir, writable=True) as ctx:
        load_dox(dox_py_fn)

        # Execute experiments
        ctx.run()

    assert len(ctx.store) == 2, "Trials: {}".format(", ".join(ctx.store.keys()))


@pytest.fixture(name="unknown_fn")
def fixture_unknown_fn(tmp_path):
    fn = str(tmp_path / "unknown.txt")
    with open(fn, "w"):
        pass

    return fn


def test_unknown_extension(unknown_fn):
    wdir = os.path.splitext(unknown_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir):
        with pytest.raises(DOXError):
            load_dox(unknown_fn)


@pytest.fixture(name="malformed_py_fn")
def fixture_malformed_py_fn(tmp_path):
    fn = str(tmp_path / "malformed.py")
    with open(fn, "w") as f:
        f.write("This is not a python file!")

    return fn


def test_malformed_py(malformed_py_fn):
    wdir = os.path.splitext(malformed_py_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir):
        with pytest.raises(DOXError):
            load_dox(malformed_py_fn)
