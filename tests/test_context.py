import pytest

from experitur.core.context import Context, DependencyError, get_current_context
from experitur.core.experiment import Experiment
from experitur.parameters import Grid


def test_Context_enter():
    """Make sure that the Context context manager works in the expected way."""
    with Context() as outer_ctx:
        with Context() as inner_ctx:
            assert get_current_context() == inner_ctx

        assert get_current_context() == outer_ctx


def test__order_experiments_fail(tmp_path):
    with Context(str(tmp_path)) as ctx:
        # Create a dependency circle
        a = Experiment("a")
        b = Experiment("b", parent=a)
        a.parent = b

        with pytest.raises(DependencyError):
            ctx.run()


def test_dependencies(tmp_path):
    with Context(str(tmp_path)) as ctx:

        @Experiment("a")
        def a(trial):
            pass

        b = Experiment("b", parent=a)

        ctx.run([b])


def test_get_experiment(tmp_path):
    with Context(str(tmp_path)) as ctx:

        @Experiment("a")
        def a(trial):
            pass

    ctx.get_experiment("a")

    with pytest.raises(KeyError):
        ctx.get_experiment("inexistent")


def test_merge_config(tmp_path):
    config = {
        k: not v for k, v in Context._default_config.items() if isinstance(v, bool)
    }

    config["a"] = 1
    config["b"] = 2
    config["c"] = 3

    with Context(str(tmp_path), config=config.copy()) as ctx:
        assert all(v == ctx.config[k] for k, v in config.items())


def test_collect(tmp_path):
    with Context(str(tmp_path)) as ctx:

        @Grid({"a": [1, 2, 3], "b": [1, 2, 3]})
        @Experiment()
        def experiment(parameters):
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
        "resolved_parameters.b",
        "result.a",
        "parameters.b",
        "experiment.func",
        "resolved_parameters.a",
        "experiment.name",
        "result.b",
        "parameters.a",
        "success",
        "experiment.independent_parameters",
    }
