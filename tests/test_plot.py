import numpy as np
import pandas as pd
import pandas.testing
import pytest

from experitur.configurators import Grid
from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial import Trial

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Only import plot if matplotlib is available
    from experitur.plot import (
        Categorical,
        Dimension,
        Integer,
        Numeric,
        Real,
        Space,
        plot_partial_dependence,
    )
except ImportError as exc:
    pytestmark = pytest.mark.skip(str(exc))


def test_Categorical():
    categorical = Categorical(categories=["a", "b", ""])
    assert categorical.transformed_size == 3

    # Linspace is independent of supplied argument
    ls3 = categorical.linspace(3)
    assert ls3.shape == (3,)
    assert list(ls3) == ["a", "b", ""]

    # Even if more items are requested, only the existing categories are returned.
    # This is to avoid duplicated output which leads to zig-zaggy plots.
    ls4 = categorical.linspace(4)
    assert ls4.shape == (3,)
    assert list(ls4) == ["a", "b", ""]

    np.testing.assert_equal(categorical.transform(["a"]), np.array([[1, 0, 0]]))

    np.testing.assert_equal(
        categorical.transform(ls4), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )

    assert categorical.rvs_transformed(4).shape == (4, 3)


def test_CategoricalBinary():
    # Binary Categoricals have a transformed_size of 1
    categorical = Categorical(categories=["a", "b"])
    assert categorical.transformed_size == 1

    # Linspace is independent of supplied argument
    ls2 = categorical.linspace(2)
    assert ls2.shape == (2,)
    assert list(ls2) == ["a", "b"]

    # Even if more items are requested, only the existing categories are returned.
    # This is to avoid duplicated output which leads to zig-zaggy plots.
    ls3 = categorical.linspace(3)
    assert ls3.shape == (2,)
    assert list(ls3) == ["a", "b"]

    np.testing.assert_equal(categorical.transform(["a"]), np.array([[0]]))

    np.testing.assert_equal(
        categorical.transform(ls3), np.array([[0], [1]]),
    )

    assert categorical.rvs_transformed(4).shape == (4, 1)


def test_Space():
    space = Space(
        [
            Integer(0, 10, label="Integer"),
            Real(0, 1, label="Real"),
            Categorical(["a", "b", "c"], label="Categorical"),
        ]
    )

    n_dim = sum(dim.transformed_size for dim in space.dimensions)
    assert n_dim == 5

    transformed = space.transform([5], [0.5], "a")

    assert transformed.shape == (1, 5)

    samples = space.rvs_transformed(3)
    assert samples.shape == (3, 5)


def test_Space2():
    space = Space([Categorical(categories=["a", "b"],), Real(0, 1),])

    samples = space.rvs_transformed(3)
    assert samples.shape == (3, 2)


def test_Dimension_get_instance_labels():
    d = Dimension.get_instance("foo", np.array([1, 2, 3]), "Foo")

    assert isinstance(d, Integer)
    assert d.label == "Foo"


# deselect with '-m "not slow"'
@pytest.mark.slow
def test_plot_partial_dependence(tmp_path):
    print(tmp_path)
    with Context(
        str(tmp_path), config={"catch_exceptions": False}, writable=True
    ) as ctx:

        @Grid(
            {
                "categorical": [1, None, "a", (1, 2), [1, 2]],
                "integer": list(range(3)),
                "float": [i / 3.0 for i in range(3)],
            }
        )
        @Experiment()
        def experiment(trial: Trial):
            assert trial.wdir

            y = trial["float"] + trial["integer"]

            if trial["categorical"] == 1:
                y *= 1
            elif trial["categorical"] == None:
                y *= 2
            elif trial["categorical"] == "a":
                y *= 3
            elif trial["categorical"] == (1, 2):
                y *= 4
            elif trial["categorical"] == [1, 2]:
                y *= 5

            return {"y": y}

    ctx.run()

    plot_partial_dependence(experiment.trials, "y", maximize=True)

    plt.savefig(tmp_path / "plot.pdf")

    print(tmp_path)


def test_Numeric_log10():
    logfloat = [10 ** -i for i in range(4)]

    n = Numeric(scale="log10")
    n.initialize("logfloat", logfloat)

    prepared = n.prepare(logfloat)

    pandas.testing.assert_series_equal(
        prepared,
        pd.Series([-i for i in range(4)]),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_plot_partial_dependence_scale_log10(tmp_path):

    with Context(str(tmp_path), writable=True) as ctx:

        @Grid(
            {"integer": list(range(4)), "logfloat": [10 ** -i for i in range(4)],}
        )
        @Experiment()
        def experiment(parameters):  # pylint: disable=unused-variable
            y = parameters["integer"] * parameters["logfloat"]

            return {"y": y}

    ctx.run()

    plot_partial_dependence(
        experiment.trials,
        "y",
        dimensions={"integer": Integer(), "logfloat": Numeric(scale="log10")},
        maximize=True,
    )

    fig_fn = tmp_path / "plot.pdf"

    plt.savefig(fig_fn)

    print(fig_fn)


@pytest.mark.slow
def test_plot_partial_dependence_scale_custom(tmp_path):

    with Context(str(tmp_path), writable=True) as ctx:

        @Grid(
            {"integer": list(range(4)), "logfloat": [1 - 10 ** -i for i in range(4)],}
        )
        @Experiment()
        def experiment(parameters):  # pylint: disable=unused-variable
            y = parameters["integer"] * parameters["logfloat"]

            return {"y": y}

    ctx.run()

    import matplotlib.pyplot as plt

    plot_partial_dependence(
        experiment.trials,
        "y",
        dimensions={
            "integer": Integer(),
            "logfloat": Integer(
                scale=lambda x: -np.log10(1 - x),
                formatter=FuncFormatter(lambda y, p: f"{1-10**-y}"),
            ),
        },
        maximize=True,
    )

    fig_fn = tmp_path / "plot.pdf"

    plt.savefig(fig_fn)

    print(fig_fn)

