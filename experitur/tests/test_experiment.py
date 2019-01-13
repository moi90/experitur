from experitur.experiment import Experiment, ExperimentError
import os
import pytest
import inspect


@pytest.fixture(name="malformed_yaml")
def fixture_malformed_yaml(tmp_path):
    fn = tmp_path / "malformed_yaml.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        [
        ---
        # Malformed YAML
        """))

    return fn


@pytest.fixture(name="no_yaml")
def fixture_no_yaml(tmp_path):
    fn = tmp_path / "malformed_yaml.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        # No YAML
        """))

    return fn


@pytest.fixture(name="empty_yaml")
def fixture_empty_yaml(tmp_path):
    fn = tmp_path / "malformed_yaml.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        ---
        # Empty YAML
        """))

    return fn


def test_malformed_yaml(malformed_yaml):
    with pytest.raises(ExperimentError):
        exp = Experiment(malformed_yaml)


def test_no_yaml(no_yaml):
    with pytest.raises(ExperimentError):
        exp = Experiment(no_yaml)


def test_empty_yaml(empty_yaml):
    with pytest.raises(ExperimentError):
        exp = Experiment(empty_yaml)


def test_experiment():
    exp = Experiment(os.path.join(
        os.path.dirname(__file__), "test_example.md"))

    exp.run()
