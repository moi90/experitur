import time

from experitur import Experiment
from experitur.configurators import Grid


@Grid({"a": [1, 2], "b": [3, 4]})
@Experiment()
def simple(trial):
    print("a:", trial["a"])
    print("b:", trial["b"])

    time.sleep(0.5)
