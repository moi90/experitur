import functools
import glob
import os.path
import re

import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial import (
    FileTrialStore,
    Trial,
    TrialParameters,
    _callable_to_name,
    _format_independent_parameters,
    _match_parameters,
)


def noop():
    pass


def test__callable_to_name():
    assert _callable_to_name(noop) == "test_trial.noop"
    assert _callable_to_name([noop]) == ["test_trial.noop"]
    assert _callable_to_name((noop,)) == ("test_trial.noop",)
    assert _callable_to_name({"noop": noop}) == {"noop": "test_trial.noop"}


def test__match_parameters():
    assert _match_parameters({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 4, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})


def test__format_independent_parameters():
    parameters = {"a": 1, "b": 2}
    assert _format_independent_parameters(parameters, []) == "_"
    assert _format_independent_parameters(parameters, ["a", "b"]) == "a-1_b-2"


def test_trial_store(tmp_path):
    with Context(str(tmp_path)) as ctx:
        ctx: Context
        with FileTrialStore(ctx) as trial_store:

            def test(trial):
                return {"result": (1, 2)}

            experiment = Experiment("test", parameters={"a": [1, 2], "b": [2, 3]})(test)

            def test2(trial):
                return {"result": (2, 4)}

            experiment2 = Experiment("test2", parent=experiment)(test2)

            trial_store["foo"] = Trial(trial_store, data={1: "foo", "bar": 2})
            assert trial_store["foo"].data == {1: "foo", "bar": 2}

            trial_store["bar/baz"] = Trial(trial_store, data={1: "foo", "bar": 2})
            assert trial_store["bar/baz"].data == {1: "foo", "bar": 2}

            trial = trial_store.create({"parameters": {"a": 1, "b": 2}}, experiment)

            fake_folder = os.path.join(ctx.wdir, trial_store.PATTERN.format("fake"))

            os.makedirs(fake_folder, exist_ok=True)

            assert trial.data["id"] == "test/a-1_b-2"

            assert "test/a-1_b-2" in trial_store

            assert len(trial_store) == 3

            del trial_store["bar/baz"]
            del trial_store["foo"]

            with pytest.raises(KeyError):
                del trial_store["foo"]

            trial_store.create({"parameters": {"a": 2, "b": 3}}, experiment)
            trial_store.create({"parameters": {"a": 3, "b": 4}}, experiment)
            trial_store.create({"parameters": {"a": 3, "b": 4}}, experiment2)

            assert set(trial_store.match(func=test).keys()) == {
                "test/a-1_b-2",
                "test/a-2_b-3",
                "test/a-3_b-4",
            }

            assert set(trial_store.match(parameters={"a": 1, "b": 2}).keys()) == {
                "test/a-1_b-2"
            }

            assert set(trial_store.match(experiment="test2")) == {"test2/a-3_b-4"}

            result = trial.run()

            assert result == {"result": (1, 2)}

            # Write trial data back to the store
            trial.save()

            assert trial_store["test/a-1_b-2"].data["result"] == {"result": (1, 2)}

            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 1}}, experiment)

            experiment.parameter_generator.generators[0].grid["c"] = [1, 2, 3]

            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)
            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)


def test_trial(tmp_path):
    with Context(str(tmp_path)) as ctx:
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


def test_trial_parameters(tmp_path, recwarn):
    config = {"catch_exceptions": False}
    with Context(str(tmp_path), config) as ctx:

        @Experiment(parameters={"a": [1], "b": [2], "c": ["{a}"]})
        def experiment(parameters: TrialParameters):
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
                parameters.prefixed("__empty_").call(identity)

        ctx.run()
