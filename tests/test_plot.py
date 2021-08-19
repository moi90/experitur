from experitur.core.trial import Trial
import pytest

from experitur.configurators import Grid
from experitur.core.context import Context
from experitur.core.experiment import Experiment

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
    import pandas.testing

    from experitur.plot import Integer, Numeric, plot_partial_dependence
except ImportError as exc:
    pytestmark = pytest.mark.skip(str(exc))


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
    n.init("logfloat", logfloat)

    prepared = n.prepare(logfloat)

    pandas.testing.assert_series_equal(
        prepared,
        pd.Series([-i for i in range(4)]),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_plot_partial_dependence_scale(tmp_path):

    with Context(str(tmp_path), writable=True) as ctx:

        @Grid(
            {
                "integer": list(range(4)),
                "logfloat": [10 ** -i for i in range(4)],
            }
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
