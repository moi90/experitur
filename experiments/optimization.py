import math
import operator

import click
import matplotlib.pyplot as plt
import pandas as pd
import skopt
from scipy.stats import uniform
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Categorical, Dimension, Integer, Real
from skopt.utils import dimensions_aslist, point_asdict, point_aslist

from experitur import Experiment, Trial
from experitur.parameters import Grid, Multi, SKOpt
from experitur.parameters.skopt import convert_trial


def rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


@Grid({"a": [1], "b": [100]})
@SKOpt({"x": Real(-10, 10), "y": Real(-10, 10)}, "z", 200)
@Grid({"repetition": [1, 2, 3]})
@Experiment(active=False,)
def exp(trial: Trial):
    z = trial.call(rosenbrock)

    print(z)

    return {"z": z}


def hyper(x, y):
    return x ** 2 + y ** 2


@Experiment(
    parameters=SKOpt({"x": Real(-10, 10), "y": Real(-10, 10)}, "z", 100), active=False
)
def exp2(trial: Trial):
    z = trial.call(hyper)
    return {"z": z}


def fun(x, y, z):
    return x ** 2 + math.sin(y) + z


@Experiment(
    parameters=SKOpt(
        {"x": Real(-10, 10), "y": Real(-10, 10), "z": Real(-10, 10)}, "res", 100
    )
)
def exp3(trial: Trial):
    res = trial.call(fun)
    return {"res": res}


def as_dimension(d):
    if isinstance(d, Dimension):
        return d
    if isinstance(d, list):
        return Categorical(d)
    raise ValueError(f"Unexpected dimension: {d!r}")


@click.argument("objective")
@click.option("--plot-objective", "plot_objective_", is_flag=True)
@click.option("--plot-convergence", "plot_convergence_", is_flag=True)
@exp3.command(target="experiment")
def show(
    experiment: Experiment, objective, plot_objective_=False, plot_convergence_=False
):
    """
    Show information about optimization.

    Run `experitur do optimization.py show exp3 res` to show information about the optimization that takes place in `exp3` regarding objective `res`.
    """

    search_space = {
        n: as_dimension(d)
        for n, d in experiment.parameter_generator.varying_parameters.items()
    }

    trials = experiment.ctx.store.match(experiment=experiment)

    results = [
        convert_trial(trial, search_space, objective)
        for trial in sorted(trials.values(), key=lambda trial: trial.data["time_start"])
    ]

    if not results:
        print("No results.")
        return

    X, Y = zip(*results)

    optimizer = skopt.Optimizer(dimensions_aslist(search_space))

    optimize_result = optimizer.tell(X, Y)

    print(
        f"Current optimium {optimize_result.fun} at {point_asdict(search_space, optimize_result.x)}"
    )

    if plot_objective_:
        plot_objective(optimize_result, levels=20)
        plt.show()
    elif plot_convergence_:
        plot_convergence(optimize_result)
        plt.show()
    else:
        print(
            "Use --plot-convergence to plot the convergence trace or --plot-objective to show the pairwise dependence plot of the objective function."
        )
