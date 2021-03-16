import pytest
import yaml

from experitur.helpers.dumper import ExperiturDumper

try:
    import numpy as np
except ImportError:
    np = None


@pytest.mark.skipif(np is None, reason="requires numpy")
def test_ndarray_representer():
    x = np.array([[1.0, 2.0, 3.0]])
    s = yaml.dump({"x": x}, Dumper=ExperiturDumper)
    print(f"{x!r} -> {s!r}")


@pytest.mark.skipif(np is None, reason="requires numpy")
def test_number_representer():
    x = np.uint8([1, 2, 3])[0]
    yaml.dump(x)
    s = yaml.dump({"x": x}, Dumper=ExperiturDumper)
    print(f"{x!r} -> {s!r}")
