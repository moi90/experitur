import inspect
import os

import pandas as pd
import pytest
from experitur.configurators import Grid
from experitur.core.context import (
    Context,
    ContextError,
    DependencyError,
    get_current_context,
)
from experitur.core.experiment import Experiment


def test_Context_enter():
    """Make sure that the Context context manager works in the expected way."""
    with Context() as outer_ctx:
        with Context() as inner_ctx:
            assert get_current_context() == inner_ctx

        assert get_current_context() == outer_ctx


def test__order_experiments_fail(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:
        # Create a dependency circle
        a = Experiment("a")
        b = Experiment("b", depends_on=a)
        a.add_dependency(b)

        with pytest.raises(DependencyError):
            ctx.run()


def test_dependencies(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment("a")
        def a(_):
            pass

        b = Experiment("b", parent=a)

        ctx.run([b])


def test_get_experiment(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Experiment("a")
        def a(_):  # pylint: disable=unused-variable
            pass

    ctx.get_experiment("a")

    with pytest.raises(KeyError):
        ctx.get_experiment("inexistent")


def test_merge_config(tmp_path):
    config = {
        k: not v
        for k, v in Context._default_config.items()  # pylint: disable=protected-access
        if isinstance(v, bool)
    }

    config["a"] = 1
    config["b"] = 2
    config["c"] = 3

    with Context(str(tmp_path), config=config.copy()) as ctx:
        assert all(v == ctx.config[k] for k, v in config.items())


def test_collect(tmp_path):
    pytest.importorskip("pandas")

    with Context(str(tmp_path), writable=True) as ctx:

        @Grid({"a": [1, 2, 3], "b": [1, 2, 3]})
        @Experiment()
        def experiment(parameters):  # pylint: disable=unused-variable
            return dict(parameters)

    ctx.run()

    result_fn = tmp_path / "result.csv"

    ctx.collect(result_fn)

    import pandas as pd

    results = pd.read_csv(result_fn, index_col=None)

    columns = set(results.columns)

    assert columns == {
        "id",
        "wdir",
        "time_start",
        "time_end",
        "experiment.meta",
        "experiment.parent",
        "experiment.independent_parameters",
        "experiment.varying_parameters",
        "experiment.minimize",
        "experiment.maximize",
        "resolved_parameters.b",
        "result.a",
        "parameters.b",
        "experiment.func",
        "resolved_parameters.a",
        "experiment.name",
        "result.b",
        "parameters.a",
        "success",
        "error",
        "used_parameters",
        "unused_parameters",
        "tags",
        "revision",
        "used_parameters",
    }


def test_readonly(tmp_path):
    with Context(str(tmp_path), writable=False) as ctx:

        @Experiment()
        def experiment(parameters):  # pylint: disable=unused-variable
            return dict(parameters)

    with pytest.raises(ContextError):
        ctx.run()


def test_stop(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Grid({"a": [1, 2, 3]})
        @Experiment()
        def experiment(trial):  # pylint: disable=unused-variable
            ctx.stop()
            return dict(trial)

        ctx.run()

        assert len(ctx.trials.match(experiment=experiment)) == 1


@pytest.fixture(name="dox_py_fn")
def fixture_dox_py_fn(tmp_path):
    fn = str(tmp_path / "dox.py")
    with open(fn, "w") as f:
        f.write(
            inspect.cleandoc(
                """
                from experitur import Experiment

                @Experiment(
                    configurator={
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

                # This experiment shouldn't be executed, because this combination of callable and parameters was already executed.
                third_experiment = Experiment(
                    parent=baseline
                )
                """
            )
        )

    return fn


@pytest.mark.xfail(strict=True)
def test_dox_py(dox_py_fn):
    wdir = os.path.splitext(dox_py_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir, writable=True) as ctx:
        ctx.load_experiments(dox_py_fn)

        # Make sure that the experiment name is guessed from the module
        ctx.get_experiment("third_experiment")

        # Execute experiments
        ctx.run()

    # This fails currently, because of the [resolved_]parameters and RecursiveDict mess.
    assert (
        len(ctx.store) == 2
    ), "Trials: {}. Expected only baseline to have been executed.".format(
        ", ".join(ctx.store.keys())
    )


@pytest.fixture(name="unknown_fn")
def fixture_unknown_fn(tmp_path):
    fn = str(tmp_path / "unknown.txt")
    with open(fn, "w"):
        pass

    return fn


def test_unknown_extension(unknown_fn):
    wdir = os.path.splitext(unknown_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir) as ctx:
        with pytest.raises(ImportError):
            ctx.load_experiments(unknown_fn)


@pytest.fixture(name="malformed_py_fn")
def fixture_malformed_py_fn(tmp_path):
    fn = str(tmp_path / "malformed.py")
    with open(fn, "w") as f:
        f.write("This is not a python file!")

    return fn


def test_malformed_py(malformed_py_fn):
    wdir = os.path.splitext(malformed_py_fn)[0]
    os.makedirs(wdir, exist_ok=True)

    with Context(wdir) as ctx:
        with pytest.raises(ImportError):
            ctx.load_experiments(malformed_py_fn)
