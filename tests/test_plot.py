import matplotlib
import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.parameters import Grid
from experitur.plot import plot_partial_dependence

matplotlib.use("Agg")


@pytest.mark.slow
def test_plot_partial_dependence(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:

        @Grid(
            {
                "categorical": [1, None, "a", (1, 2), [1, 2]],
                "integer": list(range(3)),
                "float": [i / 3.0 for i in range(3)],
            }
        )
        @Experiment()
        def experiment(parameters):  # pylint: disable=unused-variable
            y = parameters["float"] + parameters["integer"]

            if parameters["categorical"] == 1:
                y *= 1
            elif parameters["categorical"] == None:
                y *= 2
            elif parameters["categorical"] == "a":
                y *= 3
            elif parameters["categorical"] == (1, 2):
                y *= 4
            elif parameters["categorical"] == [1, 2]:
                y *= 5

            return {"y": y}

    ctx.run()

    plot_partial_dependence(experiment.trials, "y", maximize=True)

    import matplotlib.pyplot as plt

    plt.savefig(tmp_path / "plot.pdf")

    print(tmp_path)
