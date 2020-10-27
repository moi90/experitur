import functools
import re

import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial import Trial
from experitur.util import callable_to_name


def noop():
    pass


def test_callable_to_name():
    assert callable_to_name(noop) == "test_trial.noop"
    assert callable_to_name([noop]) == ["test_trial.noop"]
    assert callable_to_name((noop,)) == ("test_trial.noop",)
    assert callable_to_name({"noop": noop}) == {"noop": "test_trial.noop"}


def test_trial(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:
        # Dummy function
        def parametrized(a=1, b=2, c=3, d=4):
            return dict(a=a, b=b, c=c, d=d)

        @Experiment(parameters={"a": [1], "b": [2]})
        def experiment1(parameters):
            print("trial.wdir:", parameters.wdir)

            parameters.prefixed("parametrized_").record_defaults(parametrized, d=5)

            return parameters.prefixed("parametrized_").call(parametrized)

        ctx.run()

        trial1 = ctx.get_trials(experiment=experiment1).one()
        assert trial1.result == {"a": 1, "b": 2, "c": 3, "d": 5}

        assert len(ctx.store) == 1
        trial1.remove()
        assert len(ctx.store) == 0


def test_trial_parameters(tmp_path):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(parameters={"a": [1], "b": [2], "c": ["{a}"]})
        def experiment(trial: Trial):
            assert ctx.current_trial == trial

            assert trial["a"] == 1
            assert trial["b"] == 2
            assert trial["c"] == trial["a"]
            assert len(trial) == 3

            print(trial["a"], trial["c"])

            for k, v in trial.items():
                pass

            trial["a"] = 0
            trial["b"] = 0

            del trial["a"]
            del trial["b"]

            with pytest.raises(KeyError):
                _ = trial["a"]

            with pytest.raises(KeyError):
                _ = trial["b"]

            with pytest.raises(AttributeError):
                _ = trial.inexisting_attribute

            # test .prefixed
            seed = {"a": 1, "b": 2, "c": 3}
            for k, v in seed.items():
                trial["prefix__" + k] = v
                trial["prefix1__" + k] = v

            assert trial.prefixed("prefix__") == seed

            # test call
            def identity(a, b, c=4, d=5):
                return (a, b, c, d)

            assert trial.prefixed("prefix__").call(identity) == (1, 2, 3, 5)

            # test call: keyword parameter
            assert trial.prefixed("prefix1__").call(identity, c=6, d=7) == (1, 2, 3, 7,)
            assert trial["prefix1__d"] == 7

            # test record_defaults
            trial.prefixed("prefix2__").record_defaults(identity)
            assert trial.prefixed("prefix2__") == {"c": 4, "d": 5}

            # test call: functools.partial

            # Positional arguments will not be recorded and can't be overwritten
            identity_a8 = functools.partial(identity, 8)
            assert trial.prefixed("prefix3__").call(identity_a8, b=2) == (8, 2, 4, 5,)
            assert "prefix3__a" not in trial

            with pytest.raises(TypeError):
                trial.prefixed("prefix3__").call(identity_a8, a=9, b=2)
            assert "prefix3__a" not in trial

            # Keyword arguments will *not* be recorded and can *not* be overwritten
            identity_a8_kwd = functools.partial(identity, a=8)
            trial.prefixed("prefix4_").record_defaults(identity_a8_kwd)
            assert trial.prefixed("prefix4_") == {"c": 4, "d": 5}

            with pytest.raises(TypeError):
                trial.prefixed("prefix5_").record_defaults(identity_a8_kwd, a=9)

            assert trial.prefixed("prefix6__").call(identity_a8_kwd, b=2) == (
                8,
                2,
                4,
                5,
            )
            assert "prefix6__a" not in trial

            # Keyword arguments will be not recorded and can not be overwritten
            identity_d9_kwd = functools.partial(identity, d=9)
            assert trial.prefixed("prefix7__").call(identity_d9_kwd, 1, 2) == (
                1,
                2,
                4,
                9,
            )
            assert trial.prefixed("prefix7__") == {"c": 4}

            with pytest.raises(TypeError):
                trial.prefixed("prefix7__").call(identity_d9_kwd, 1, 2, 5, 10)

            # Ignore superfluous parameter when already set by partial
            trial.prefixed("prefix7__")["d"] = 10
            assert trial.prefixed("prefix7__").call(identity_d9_kwd, 1, 2) == (
                1,
                2,
                4,
                9,
            )

            ### Partial end

            # setdefaults
            assert trial.prefixed("prefix8__").setdefaults(
                dict(a=1, b=2, c=3, d=4), e=10
            ) == dict(a=1, b=2, c=3, d=4, e=10)
            assert dict(trial.prefixed("prefix8__")) == dict(a=1, b=2, c=3, d=4, e=10)

            # Make sure that the default value is recorded when using .get
            assert trial.prefixed("prefix7__").get("f", 10) == 10
            assert trial.prefixed("prefix7__")["f"] == 10

            # Missing parameters
            with pytest.raises(
                TypeError, match=re.escape("Missing required parameter(s) 'a', 'b'")
            ):
                trial.prefixed("__empty1_").call(identity)

            def fun(*_, **__):
                pass

            trial.prefixed("__empty1b_").call(fun)

            ### parameters.choice
            class A:
                pass

            def b():
                pass

            class C:
                pass

            c = C()

            assert (
                trial.prefixed("__empty2_").choice("parameter_name", [A, b, c], "A")
                == A
            )

            trial.prefixed("__empty2_")["parameter_name"] = "b"

            assert (
                # pylint: disable=comparison-with-callable
                trial.prefixed("__empty2_").choice("parameter_name", [A, b, c], "A")
                == b
            )

            trial.prefixed("__empty2_")["parameter_name"] = "C"

            assert (
                trial.prefixed("__empty2_").choice("parameter_name", [A, b, c], "A")
                == c
            )

            x, y = 1, 2

            with pytest.raises(ValueError):
                trial.prefixed("__empty2_").choice("parameter_name", [x, y], "a")

            assert (
                trial.prefixed("__empty3_").choice(
                    "parameter_name", {"x": x, "y": y}, "x"
                )
                == x
            )

            with pytest.raises(ValueError):
                trial.prefixed("__empty3_").choice("parameter_name", [A, A], "A")

            with pytest.raises(ValueError):
                trial.prefixed("__empty3_").choice("parameter_name", A, "A")  # type: ignore

        ctx.run()

        assert ctx.current_trial is None


def test_trial_logging(tmp_path):
    config = {"catch_exceptions": False}

    print(tmp_path)

    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment()
        def experiment(trial: Trial):  # pylint: disable=unused-variable
            for i in range(10):
                trial.log({"i": i, "i10": i * 10}, ni=1 / (i + 1))

    ctx.run()

    trial = ctx.get_trials().one()

    log_entries = trial._logger.read()
    assert log_entries == [
        {"i": i, "i10": i * 10, "ni": 1 / (i + 1)} for i in range(10)
    ]
