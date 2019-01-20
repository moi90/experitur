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
    fn = tmp_path / "no_yaml.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        # No YAML
        """))

    return fn


@pytest.fixture(name="empty_yaml")
def fixture_empty_yaml(tmp_path):
    fn = tmp_path / "empty_yaml.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        ---
        # Empty YAML
        """))

    return fn


@pytest.fixture(name="no_list_dict")
def fixture_no_list_dict(tmp_path):
    fn = tmp_path / "no_list_dict.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        foo
        ---
        # Neither list nor dict
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


def test_no_list_dict(no_list_dict):
    with pytest.raises(ExperimentError):
        exp = Experiment(no_list_dict)


@pytest.fixture(name="dict_simple")
def fixture_dict_simple(tmp_path):
    fn = tmp_path / "dict_simple.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        id: dict_simple
        ---
        # Dict simple
        """))

    return fn


def test_dict_simple(dict_simple):
    exp = Experiment(dict_simple)


@pytest.fixture(name="list_base")
def fixture_list_base(tmp_path):
    fn = tmp_path / "list_base.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        -
                id: baseline
        -
                id: derived
                base: baseline
        -
                id: derived2
                base: derived
        ---
        # Base
        """))

    return fn


def test_list_base(list_base):
    exp = Experiment(list_base)
    exp.run()


@pytest.fixture(name="list_base_notfound")
def fixture_list_base_notfound(tmp_path):
    fn = tmp_path / "list_base_notfound.md"
    with open(fn, "w") as f:
        f.write(inspect.cleandoc("""
        ---
        -
                id: derived
                base: baseline
        ---
        # list_base_notfound
        """))

    return fn


def test_list_base_notfound(list_base_notfound):
    exp = Experiment(list_base_notfound)

    with pytest.raises(ExperimentError):
        exp.run()


def test_experiment():
    exp = Experiment(os.path.join(
        os.path.dirname(__file__), "test_example.md"))

    exp.run()