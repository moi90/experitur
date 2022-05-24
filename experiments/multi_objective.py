from experitur import Experiment, Trial
from experitur.parameters import SKOpt


@SKOpt({"x": (0, 5.0), "y": (0, 3.0)}, 20, minimize=["y1", "y2"], n_initial_points=1)
@Experiment()
def multi_objective(trial: Trial):
    """
    Binh and Korn function.

    Binh T. and Korn U. (1997) MOBES: A Multiobjective Evolution Strategy for Constrained Optimization Problems.
    In: Proceedings of the Third International Conference on Genetic Algorithms. Czech Republic. pp. 176–182
    """
    return {
        "y1": 4 * trial["x"] ** 2 + 4 * trial["y"] ** 2,
        "y2": (trial["x"] - 5) ** 2 + (trial["y"] - 5) ** 2,
    }
