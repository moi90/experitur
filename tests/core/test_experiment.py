import pytest

from experitur.core.context import Context, ContextError
from experitur.core.experiment import (
    Experiment,
    ExperimentError,
    format_trial_parameters,
    _detect_keyboard_interrupt,
)
from experitur.core.configurators import Configurator, Const, Grid
from experitur.core.trial import Trial
from experitur import unset


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

        configurator = Grid({"a": [1, 2]})

        @Experiment(
            "a",
            configurator=configurator,
            meta={"a": "foo"},
            maximize="a",
            minimize="b",
        )
        def a(_):
            return {"a": 1, "b": 2}

        b = Experiment("b", parent=a, maximize="b")

        # Ensure that meta is *copied* to child experiments
        assert id(b.meta) != id(a.meta)
        assert b.meta == a.meta

        # If maximize or minimize are specified, previous optimization settings are overwritten.
        assert b.maximize == ["b"]

        # Assert that inherited samplers are concatenated in the right way
        c = Experiment("c", configurator=Grid({"b": [1, 2]}), parent=a)

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

        @Experiment(configurator={"a": [1, 2, 3]})
        def exp2(_):  # pylint: disable=unused-variable
            pass

        @Experiment(configurator=[{"a": [1, 2, 3]}])
        def exp3(_):  # pylint: disable=unused-variable
            pass

        @Experiment(configurator=Grid({}))
        def exp4(_):  # pylint: disable=unused-variable
            pass

        with pytest.raises(ValueError):

            @Experiment(configurator=1)
            def exp5(_):  # pylint: disable=unused-variable
                pass


def test_configurator_order(tmp_path):
    class ConcreteConfigurator(Configurator):
        @property
        def parameter_values(self):
            return {}

    class PG1(ConcreteConfigurator):
        pass

    class PG2(ConcreteConfigurator):
        pass

    class PG3(ConcreteConfigurator):
        pass

    class PG4(ConcreteConfigurator):
        pass

    class PG5(ConcreteConfigurator):
        pass

    with Context(str(tmp_path)):

        @PG1()
        @PG2()
        @Experiment()
        def parent_experiment(_):
            pass

        @PG3()
        @PG4()
        @Experiment(configurator=PG5(), parent=parent_experiment)
        def child_experiment(_):
            pass

        pg_types = [
            type(pg)
            for pg in child_experiment._configurators  # pylint: disable=protected-access
        ]

        assert pg_types == [PG1, PG2, PG3, PG4, PG5]


def test_successful_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment()
        def experiment(_):  # pylint: disable=unused-variable
            pass

        on_success_called = 0

        @experiment.on_success
        def _(_):
            nonlocal on_success_called
            on_success_called += 1

        ctx.run()

        assert on_success_called

        trial = ctx.trials.one()
        assert trial.error is None
        assert trial.success


def test_failing_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(volatile=True)
        def experiment(_):  # pylint: disable=unused-variable
            raise Exception("Some error")

        on_success_called = 0

        @experiment.on_success
        def _(_):
            nonlocal on_success_called
            on_success_called += 1

        with pytest.raises(Exception):
            ctx.run()

        assert not on_success_called

        trial = ctx.trials.one()
        assert trial.error == "Exception: Some error"
        assert trial.success is False


def test_volatile_experiment(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(volatile=True)
        def experiment(trial):
            assert ctx.current_trial is trial

        on_success_called = 0
        on_pre_run_called = 0

        @experiment.on_pre_run
        def _(trial):
            nonlocal on_pre_run_called

            assert isinstance(trial, Trial)
            on_pre_run_called += 1

        @experiment.on_success
        def _(trial):
            nonlocal on_success_called

            assert isinstance(trial, Trial)
            on_success_called += 1

        ctx.run()

        with pytest.raises(ContextError):
            ctx.current_trial

        assert on_success_called == 1
        assert on_pre_run_called == 1

        assert len(ctx.store) == 0


def test_parameter_substitution(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(configurator={"a1": [1], "a2": [2], "b": [1, 2], "a": ["{a{b}}"]})
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

        assert experiment.maximize == ["a"]

        with pytest.raises(ExperimentError):
            ctx.run()


def test_trials(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        sampler = Grid({"a": [1, 2]})

        @Experiment("a", configurator=Grid({"a": [1, 2], "b": [3, 4]}))
        def a(trial: Trial):
            return dict(trial)

    ctx.run()

    assert len(a.trials) == 4


def test_skip(tmp_path):
    config = {"skip_existing": True}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment("a", configurator=Grid({"a": [1, 2], "b": [3, 4]}))
        def a(trial: Trial):
            return dict(trial)

        Experiment("b", parent=a)

    ctx.run()

    assert len(ctx.trials) == 4


class ExpectedException(Exception):
    pass


def test__detect_keyboard_interrupt_explicit():
    exception_raised = False
    try:
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt as exc:
            raise ExpectedException() from exc
    except ExpectedException as exc:  # pylint: disable=broad-except
        exception_raised = True
        assert _detect_keyboard_interrupt(exc)

    assert exception_raised


def test__detect_keyboard_interrupt_implicit():
    exception_raised = False
    try:
        try:
            raise KeyboardInterrupt()
        except:
            raise ExpectedException()
    except ExpectedException as exc:  # pylint: disable=broad-except
        exception_raised = True
        assert _detect_keyboard_interrupt(exc)

    assert exception_raised


def test_cleanup_unset(tmp_path):
    config = {"skip_existing": False}
    with Context(str(tmp_path), config=config, writable=True) as ctx:

        @Experiment("a", configurator=Grid({"a": [1, 2], "b": [3, unset]}))
        def a(trial: Trial):
            return dict(trial)

    ctx.run()

    results = set(tuple(sorted(t.result.items())) for t in ctx.trials)
    assert results == set(
        [
            # a=1, b=3
            (("a", 1), ("b", 3)),
            # a=1, b=unset
            (("a", 1),),
            # a=2, b=3
            (("a", 2), ("b", 3)),
            # a=2, b=unset
            (("a", 2),),
        ]
    )


def test_checkpoint_resume(tmp_path):
    config = {"skip_existing": True, "resume_failed": True, "catch_exceptions": False}
    with Context(str(tmp_path), config=config, writable=True) as ctx:

        @Experiment()
        def a(trial: Trial, checkpoint=None):
            if checkpoint is None:
                checkpoint = 0

            trial.save_checkpoint(checkpoint=checkpoint + 1)
            assert len(trial.find_files("checkpoint_*.chk")) == 1

            if checkpoint < 1:
                raise ExpectedException()

            return dict(checkpoint=checkpoint)

        with pytest.raises(ExpectedException):
            a.run()

        trial = a.trials.one()
        assert trial.is_failed
        assert len(trial.find_files("checkpoint_*.chk")) == 1

        # Run a second time so that the trial is resumed at the checkpoint
        a.run()

        trial = a.trials.one()

        assert not trial.is_failed
        assert len(trial.find_files("checkpoint_*.chk")) == 1


def test_get_matching_trials(tmp_path):
    config = {"skip_existing": True, "catch_exceptions": False}
    with Context(str(tmp_path), config=config, writable=True) as ctx:

        # Run many experiments of the same kind with different parameter settings

        @Const(a=0)
        @Experiment()
        def experiment0(trial: Trial):
            return dict(trial)

        for i in range(1, 5):
            experiment0.child(f"experiment{i}", Const(a=i))

        experiment_total = experiment0.child(
            f"experiment_total", Grid({"a": [0, 1, 2, 3, 4]})
        )

        print(ctx.registered_experiments)

        ctx.run()

        # Because experiment_total defines all previous settings, all trials should match.

        matching_trial_ids = sorted(
            t.id for t in experiment_total.get_matching_trials()
        )
        all_trial_ids = sorted(t.id for t in ctx.trials)

        assert matching_trial_ids == all_trial_ids
