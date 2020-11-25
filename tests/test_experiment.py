import pytest

from experitur.core.context import Context
from experitur.core.experiment import (
    Experiment,
    ExperimentError,
    format_trial_parameters,
)
from experitur.core.parameters import Grid, ParameterGenerator
from experitur.core.trial import Trial


def test_meta(tmp_path):
    with Context(str(tmp_path), writable=True):

        @Experiment()
        def a(_):
            pass

        @Experiment(meta=dict(foo="bar"))
        def b(_):
            pass

    assert "hostname" in a.meta
    assert "hostname" in b.meta
    assert b.meta["foo"] == "bar"


def test_merge(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        sampler = Grid({"a": [1, 2]})

        @Experiment("a", parameters=sampler, meta={"a": "foo"})
        def a(_):
            pass

        b = Experiment("b", parent=a)

        # Ensure that meta is *copied* to child experiments
        assert id(b.meta) != id(a.meta)
        assert b.meta == a.meta

        # Assert that inherited samplers are concatenated in the right way
        c = Experiment("c", parameters=Grid({"b": [1, 2]}), parent=a)
        ctx.run()

        # Parameters in a and b should be the same
        a_params = set(
            tuple(t["parameters"].items()) for t in ctx.store.match(experiment=a)
        )
        b_params = set(
            tuple(t["parameters"].items()) for t in ctx.store.match(experiment=b)
        )

        assert a_params == b_params

        print(ctx.store.match())

        c_trials = ctx.store.match(experiment=c)

        assert len(c_trials) == 4

        parameter_configurations = set(tuple(t["parameters"].items()) for t in c_trials)

        # Assert exististence of all grid  cells
        assert parameter_configurations == {
            (("a", 2), ("b", 1)),
            (("a", 1), ("b", 1)),
            (("a", 2), ("b", 2)),
            (("a", 1), ("b", 2)),
        }


def test_parameters(tmp_path):
    with Context(str(tmp_path)):

        @Experiment()
        def exp1(_):  # pylint: disable=unused-variable
            pass

        @Experiment(parameters={"a": [1, 2, 3]})
        def exp2(_):  # pylint: disable=unused-variable
            pass

        @Experiment(parameters=[{"a": [1, 2, 3]}])
        def exp3(_):  # pylint: disable=unused-variable
            pass

        @Experiment(parameters=Grid({}))
        def exp4(_):  # pylint: disable=unused-variable
            pass

        with pytest.raises(ValueError):

            @Experiment(parameters=1)
            def exp5(_):  # pylint: disable=unused-variable
                pass


def test_parameter_generator_order(tmp_path):
    class ConcreteParameterGenerator(ParameterGenerator):
        @property
        def independent_parameters(self):
            return {}

    class PG1(ConcreteParameterGenerator):
        pass

    class PG2(ConcreteParameterGenerator):
        pass

    class PG3(ConcreteParameterGenerator):
        pass

    class PG4(ConcreteParameterGenerator):
        pass

    class PG5(ConcreteParameterGenerator):
        pass

    with Context(str(tmp_path)):

        @PG1()
        @PG2()
        @Experiment()
        def parent_experiment(_):
            pass

        @PG3()
        @PG4()
        @Experiment(parameters=PG5(), parent=parent_experiment)
        def child_experiment(_):
            pass

        pg_types = [
            type(pg)
            for pg in child_experiment._parameter_generators  # pylint: disable=protected-access
        ]

        assert pg_types == [PG1, PG2, PG3, PG4, PG5]


def test_failing_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(volatile=True)
        def experiment(_):  # pylint: disable=unused-variable
            raise Exception("Some error")

        with pytest.raises(Exception):
            ctx.run()

        trial = ctx.trials.one()
        assert trial.error == "Exception: Some error"
        assert trial.success is False


def test_volatile_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(volatile=True)
        def experiment(_):  # pylint: disable=unused-variable
            pass

        ctx.run()

        assert len(ctx.store) == 0


def test_parameter_substitution(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(parameters={"a1": [1], "a2": [2], "b": [1, 2], "a": ["{a{b}}"]})
        def experiment(trial):
            return dict(trial)

        ctx.run()

        valid = [
            t["resolved_parameters"]["a"] == t["resolved_parameters"]["b"]
            and isinstance(t["resolved_parameters"]["a"], int)
            for t in ctx.store.match(experiment=experiment)
        ]

        assert all(valid)


def test_format_trial_parameters():
    assert format_trial_parameters() == "_()"
    assert format_trial_parameters("foo") == "foo()"
    assert format_trial_parameters("foo", {"a": 1, "b": 2}) == "foo(a=1, b=2)"
    assert (
        format_trial_parameters("foo", {"a": 1, "b": 2}, "experiment")
        == "experiment:foo(a=1, b=2)"
    )


def test_minimize_maximize_list(tmp_path):
    config = {"skip_existing": False, "catch_exceptions": False}
    with Context(str(tmp_path), config):

        @Experiment(maximize="a", minimize=["b", "c"])
        def experiment(_):
            return {}

        assert isinstance(experiment.maximize, list)
        assert isinstance(experiment.minimize, list)


def test_minimize_maximize_exclusive(tmp_path):
    config = {"skip_existing": False, "catch_exceptions": False}
    with Context(str(tmp_path), config):

        # Check that the same metric is not at the same time marked as minimized and maximized
        with pytest.raises(ValueError):

            @Experiment(maximize="a", minimize="a")
            def experiment(_):  # pylint: disable=unused-variable
                pass


def test_mimimize_nonexisting(tmp_path):
    config = {"skip_existing": False, "catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(maximize="a")
        def experiment(_):  # pylint: disable=unused-variable
            return {}

        with pytest.raises(ExperimentError):
            ctx.run()


def test_trials(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        sampler = Grid({"a": [1, 2]})

        @Experiment("a", parameters=Grid({"a": [1, 2], "b": [3, 4]}))
        def a(trial: Trial):
            return dict(trial)

    ctx.run()

    assert len(a.trials) == 4


def test_skip(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment("a", parameters=Grid({"a": [1, 2], "b": [3, 4]}))
        def a(trial: Trial):
            return dict(trial)

        Experiment("b", parent=a)

    ctx.run()

    assert len(ctx.trials) == 4
