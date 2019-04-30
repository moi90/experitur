"""
Run with:
    
    experitur run py_example
"""
from pprint import pformat

from experitur import experiment, run


@experiment(
    parameter_grid={
        "a": [1, 2],
        "b": ["a", "b"]
    })
def baseline(trial):
    """This is an example experiment."""
    pformat(trial)


second_experiment = experiment(
    "second_experiment",
    parameter_grid={
        "a": [4, 5],
        "b": ["c", "d"],
        "c": [-1, -2]
    },
    parent=baseline)

if __name__ == "__main__":
    run([second_experiment])
