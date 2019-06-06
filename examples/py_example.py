"""
Run with:
    
    experitur run py_example
"""
from pprint import pformat

from experitur import experiment, run


@experiment(
    parameter_grid={
        "a1": [1],
        "a2": [2],
        "b": [1, 2],
        "a": ["{a_{b}}"],
    })
def baseline(trial):
    """This is an example experiment."""
    pformat(trial)


second_experiment = experiment(
    "second_experiment",
    parameter_grid={
        "a": [4, 5],
        "c": [-1, -2]
    },
    parent=baseline)

third_experiment = experiment(
    "third_experiment",
    parent=baseline
)

if __name__ == "__main__":
    run([second_experiment])
