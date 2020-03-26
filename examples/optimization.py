from experitur import Experiment, TrialParameters
from experitur.parameters import Grid, SKOpt


def rosenbrock_parametric(a, b, x, y):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


@Grid({"a": [1, 2], "b": [100, 101]})
@SKOpt({"x": (-10.0, 10.0), "y": (-10.0, 10.0)}, "z", 10)
@Grid({"repetition": [1, 2, 3]})
@Experiment(active=False,)
def exp(trial: TrialParameters):
    z = trial.apply("", rosenbrock_parametric)

    return {"z": z}
