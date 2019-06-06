from experitur.dox import DOX
import inspect
import pytest


@pytest.fixture(name="dox_py_fn")
def fixture_dox_py_fn(tmp_path):
    fn = str(tmp_path / "dox.py")
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        from experitur import experiment

        @experiment(
            parameter_grid={
                "a1": [1],
                "a2": [2],
                "b": [1, 2],
                "a": ["{a_{b}}"],
            })
        def baseline(trial):
            return trial.parameters

        # This experiment shouldn't be executed, because this combination of callable and parameters was already executed.
        experiment(
            "second_experiment",
            parent=baseline
        )
        """))

    return fn


def test_dox_py(dox_py_fn):
    dox = DOX(dox_py_fn)

    # Execute experiments
    dox.run()

    dox.ctx.backend.reload()

    print(dox.ctx.backend.trials())
