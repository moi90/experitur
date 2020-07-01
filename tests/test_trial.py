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
            return (a, b, c, d)

        @Experiment(parameters={"a": [1, 2], "b": [2, 3]})
        def experiment1(parameters):
            print("trial.wdir:", parameters.wdir)

            parameters.prefixed("parametrized_").record_defaults(parametrized, d=5)

            return parameters.prefixed("parametrized_").call(parametrized)

        @Experiment(parent=experiment1)
        def experiment2(trial):
            return trial.experiment1["a"]

        trial = ctx.store.create({"parameters": {"a": 1, "b": 2}}, experiment1)
        result = trial.run()
        assert result == (1, 2, 3, 5)

        trial2 = ctx.store.create({"parameters": {"a": 1, "b": 2}}, experiment2)
        result = trial2.run()
        assert result == 1

        assert len(ctx.store) == 2
        trial2.remove()
        assert len(ctx.store) == 1


def test_trial_parameters(tmp_path, recwarn):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment(parameters={"a": [1], "b": [2], "c": ["{a}"]})
        def experiment(parameters: Trial):
            assert parameters["a"] == 1
            assert parameters["b"] == 2
            assert parameters["c"] == parameters["a"]
            assert len(parameters) == 3

            print(parameters["a"], parameters["c"])

            for k, v in parameters.items():
                pass

            parameters["a"] = 0
            parameters["b"] = 0

            del parameters["a"]
            del parameters["b"]

            with pytest.raises(KeyError):
                parameters["a"]

            with pytest.raises(KeyError):
                parameters["b"]

            with pytest.raises(AttributeError):
                parameters.inexisting_attribute

            # test .prefixed
            seed = {"a": 1, "b": 2, "c": 3}
            for k, v in seed.items():
                parameters["prefix__" + k] = v
                parameters["prefix1__" + k] = v

            assert parameters.prefixed("prefix__") == seed

            # test call
            def identity(a, b, c=4, d=5):
                return (a, b, c, d)

            assert parameters.prefixed("prefix__").call(identity) == (1, 2, 3, 5)

            # test call: keyword parameter
            assert parameters.prefixed("prefix1__").call(identity, c=6, d=7) == (
                1,
                2,
                3,
                7,
            )
            assert parameters["prefix1__d"] == 7

            # test record_defaults
            parameters.prefixed("prefix2__").record_defaults(identity)
            assert parameters.prefixed("prefix2__") == {"c": 4, "d": 5}

            # test call: functools.partial

            # Positional arguments will not be recorded and can't be overwritten
            identity_a8 = functools.partial(identity, 8)
            assert parameters.prefixed("prefix3__").call(identity_a8, b=2) == (
                8,
                2,
                4,
                5,
            )
            assert "prefix3__a" not in parameters

            with pytest.raises(TypeError):
                parameters.prefixed("prefix3__").call(identity_a8, a=9, b=2)
            assert "prefix3__a" not in parameters

            # Keyword arguments will *not* be recorded and can *not* be overwritten
            identity_a8_kwd = functools.partial(identity, a=8)
            parameters.prefixed("prefix4_").record_defaults(identity_a8_kwd)
            assert parameters.prefixed("prefix4_") == {"c": 4, "d": 5}

            with pytest.raises(TypeError):
                parameters.prefixed("prefix5_").record_defaults(identity_a8_kwd, a=9)

            assert parameters.prefixed("prefix6__").call(identity_a8_kwd, b=2) == (
                8,
                2,
                4,
                5,
            )
            assert "prefix6__a" not in parameters

            # Keyword arguments will be not recorded and can not be overwritten
            identity_d9_kwd = functools.partial(identity, d=9)
            assert parameters.prefixed("prefix7__").call(identity_d9_kwd, 1, 2) == (
                1,
                2,
                4,
                9,
            )
            assert parameters.prefixed("prefix7__") == {"c": 4}

            with pytest.raises(TypeError):
                parameters.prefixed("prefix7__").call(identity_d9_kwd, 1, 2, 5, 10)

            # Ignore superfluous parameter when already set by partial
            parameters.prefixed("prefix7__")["d"] = 10
            assert parameters.prefixed("prefix7__").call(identity_d9_kwd, 1, 2) == (
                1,
                2,
                4,
                9,
            )

            ### Partial end

            # setdefaults
            assert parameters.prefixed("prefix8__").setdefaults(
                dict(a=1, b=2, c=3, d=4), e=10
            ) == dict(a=1, b=2, c=3, d=4, e=10)
            assert dict(parameters.prefixed("prefix8__")) == dict(
                a=1, b=2, c=3, d=4, e=10
            )

            # Make sure that the default value is recorded when using .get
            assert parameters.prefixed("prefix7__").get("f", 10) == 10
            assert parameters.prefixed("prefix7__")["f"] == 10

            # Missing parameters
            with pytest.raises(
                TypeError, match=re.escape("Missing required parameter(s) 'a', 'b'")
            ):
                parameters.prefixed("__empty1_").call(identity)

            def fun(*args, **kwargs):
                pass

            parameters.prefixed("__empty1b_").call(fun)

            ### parameters.choice
            class A:
                pass

            def b():
                pass

            class C:
                pass

            c = C()

            assert (
                parameters.prefixed("__empty2_").choice(
                    "parameter_name", [A, b, c], "A"
                )
                == A
            )

            parameters.prefixed("__empty2_")["parameter_name"] = "b"

            assert (
                parameters.prefixed("__empty2_").choice(
                    "parameter_name", [A, b, c], "A"
                )
                == b
            )

            parameters.prefixed("__empty2_")["parameter_name"] = "C"

            assert (
                parameters.prefixed("__empty2_").choice(
                    "parameter_name", [A, b, c], "A"
                )
                == c
            )

            x, y = 1, 2

            with pytest.raises(ValueError):
                parameters.prefixed("__empty2_").choice("parameter_name", [x, y], "a")

            assert (
                parameters.prefixed("__empty3_").choice(
                    "parameter_name", {"x": x, "y": y}, "x"
                )
                == x
            )

            with pytest.raises(ValueError):
                parameters.prefixed("__empty3_").choice("parameter_name", [A, A], "A")

            with pytest.raises(ValueError):
                parameters.prefixed("__empty3_").choice("parameter_name", A, "A")  # type: ignore

        ctx.run()


def test_trial_logging(tmp_path):
    config = {"catch_exceptions": False}

    print(tmp_path)

    with Context(str(tmp_path), config, writable=True) as ctx:

        @Experiment()
        def experiment(trial_parameters: Trial):
            for i in range(10):
                trial_parameters.log({"i": i, "i10": i * 10}, ni=1 / (i + 1))

    ctx.run()

    trial = ctx.store.match().one()

    log_entries = trial.logger.read()
    assert log_entries == [
        {"i": i, "i10": i * 10, "ni": 1 / (i + 1)} for i in range(10)
    ]
