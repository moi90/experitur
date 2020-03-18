import yaml

from experitur.helpers.dumper import ExperiturDumper
import numpy as np


def test_ndarray_representer():
    x = np.array([[1.0, 2.0, 3.0]])
    s = yaml.dump({"x": x}, Dumper=ExperiturDumper)
    print(f"{x!r} -> {s!r}")


def test_number_representer():
    x = np.uint8([1, 2, 3])[0]
    yaml.dump(x)
    s = yaml.dump({"x": x}, Dumper=ExperiturDumper)
    print(f"{x!r} -> {s!r}")
