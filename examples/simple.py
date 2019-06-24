import time

from experitur import experiment


@experiment(
    parameter_grid={
        "a": [1, 2],
        "b": [3, 4],
    })
def simple(trial):
    print("a:", trial["a"])
    print("b:", trial["b"])

    time.sleep(0.5)
