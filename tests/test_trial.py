import glob
import os.path

import pytest

from experitur.context import Context, push_context
from experitur.experiment import Experiment
from experitur.trial import (FileTrialStore, Trial, _callable_to_name,
                             _format_independent_parameters, _match_parameters)


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
    with push_context(Context(str(tmp_path))) as ctx:
        with FileTrialStore(ctx) as trial_store:

            def test(trial):
                return {"result": (1, 2)}

            experiment = ctx.experiment("test", parameter_grid={
                                        "a": [1, 2], "b": [2, 3]})(test)

            def test2(trial):
                return {"result": (2, 4)}

            experiment2 = ctx.experiment("test2", parent=experiment)(test2)

            parameters = {"a": 1, "b": 2}

            trial_store["foo"] = Trial(trial_store, data={1: "foo", "bar": 2})
            assert trial_store["foo"].data == {1: "foo", "bar": 2}

            trial_store["bar/baz"] = Trial(trial_store,
                                           data={1: "foo", "bar": 2})
            assert trial_store["bar/baz"].data == {1: "foo", "bar": 2}

            trial = trial_store.create(parameters, experiment)

            fake_folder = os.path.join(
                ctx.wdir,
                trial_store.PATTERN.format("fake"))

            os.makedirs(fake_folder, exist_ok=True)

            assert trial.data["id"] == "test/a-1_b-2"

            assert "test/a-1_b-2" in trial_store

            assert len(trial_store) == 3

            del trial_store["bar/baz"]
            del trial_store["foo"]

            with pytest.raises(KeyError):
                del trial_store["foo"]

            trial_store.create({"a": 2, "b": 3}, experiment)
            trial_store.create({"a": 3, "b": 4}, experiment)
            trial_store.create({"a": 3, "b": 4}, experiment2)

            assert set(trial_store.match(callable=test).keys()) == {
                "test/a-1_b-2", "test/a-2_b-3", "test/a-3_b-4"}

            assert set(trial_store.match(parameters={"a": 1, "b": 2}).keys()) == {
                "test/a-1_b-2"}

            assert set(trial_store.match(experiment="test2")) == {
                "test2/a-3_b-4"}

            result = trial.run()

            assert result == {"result": (1, 2)}

            # Write trial data back to the store
            trial.save()

            assert trial_store["test/a-1_b-2"].data["result"] == {
                "result": (1, 2)}

            trial_store.create({"a": 1, "b": 2, "c": 1}, experiment)

            experiment.parameter_grid["c"] = [1, 2, 3]

            trial_store.create({"a": 1, "b": 2, "c": 2}, experiment)
            trial_store.create({"a": 1, "b": 2, "c": 2}, experiment)


def test_trial(tmp_path):
    with push_context(Context(str(tmp_path))) as ctx:
        # Dummy function
        def parametrized(a=1, b=2, c=3, d=4):
            return (a, b, c, d)

        @ctx.experiment(parameter_grid={
            "a": [1, 2], "b": [2, 3]})
        def experiment1(trial):
            print("trial.wdir:", trial.wdir)

            trial.record_defaults(
                "parametrized_", parametrized, d=5)

            return trial.apply("parametrized_", parametrized)

        @ctx.experiment(parent=experiment1)
        def experiment2(trial):
            return trial.experiment1["a"]

        trial = ctx.store.create({"a": 1, "b": 2}, experiment1)
        result = trial.run()
        assert result == (1, 2, 3, 5)

        trial2 = ctx.store.create({"a": 1, "b": 2}, experiment2)
        result = trial2.run()
        assert result == 1


def test_trial_proxy(tmp_path, recwarn):
    config = {
        "catch_exceptions": False
    }
    with push_context(Context(str(tmp_path), config)) as ctx:
        @ctx.experiment(parameter_grid={
            "a": [1], "b": [2], "c": ["{a}"]})
        def experiment(trial):
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
                trial["a"]

            with pytest.raises(KeyError):
                trial["b"]

            # test without_prefix
            seed = {"a": 1, "b": 2, "c": 3}
            for k, v in seed.items():
                trial['prefix__'+k] = v

            assert trial.without_prefix('prefix__') == seed

            # test apply
            def identity(a, b, c=4, d=5):
                return (a, b, c, d)
            assert trial.apply('prefix__', identity) == (1, 2, 3, 5)

            # test apply: keyword parameter
            assert trial.apply('prefix__', identity, c=6, d=7) == (1, 2, 3, 7)

            # test record_defaults
            trial.record_defaults('prefix2__', identity, x=7)
            assert trial.without_prefix('prefix2__') == {
                "c": 4, "d": 5, "x": 7}

        ctx.run()
