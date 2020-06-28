import glob
import os.path
from typing import Type

import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial import TrialData
from experitur.core.trial_store import (
    FileTrialStore,
    TrialStore,
    _format_independent_parameters,
    _match_parameters,
)


def test__match_parameters():
    assert _match_parameters({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 4, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})


def test__format_independent_parameters():
    parameters = {"a": 1, "b": 2}
    assert _format_independent_parameters(parameters, []) == "_"
    assert _format_independent_parameters(parameters, ["a", "b"]) == "a-1_b-2"


@pytest.fixture(name="TrialStoreImplementation", params=[FileTrialStore])
def _TrialStoreImplementation(request):
    return request.param


def test_trial_store(tmp_path, TrialStoreImplementation: Type[TrialStore]):
    with Context(str(tmp_path)) as ctx:
        ctx: Context
        with TrialStoreImplementation(ctx) as trial_store:
            trial_store: TrialStore

            def test(trial):
                return {"result": (1, 2)}

            experiment = Experiment("test", parameters={"a": [1, 2], "b": [2, 3]})(test)

            def test2(trial):
                return {"result": (2, 4)}

            experiment2 = Experiment("test2", parent=experiment)(test2)

            trial_store["foo"] = TrialData(
                trial_store, data={"id": "foo", "wdir": "", 1: "foo", "bar": 2}
            )
            assert trial_store["foo"].data == {
                "id": "foo",
                "wdir": "",
                1: "foo",
                "bar": 2,
            }

            trial_store["bar/baz"] = TrialData(
                trial_store, data={"id": "bar/baz", "wdir": "", 1: "foo", "bar": 2}
            )
            assert trial_store["bar/baz"].data == {
                "id": "bar/baz",
                "wdir": "",
                1: "foo",
                "bar": 2,
            }

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

            assert set(trial.id for trial in trial_store.match(func=test)) == {
                "test/a-1_b-2",
                "test/a-2_b-3",
                "test/a-3_b-4",
            }

            assert set(
                trial.id for trial in trial_store.match(parameters={"a": 1, "b": 2})
            ) == {"test/a-1_b-2"}

            assert set(trial.id for trial in trial_store.match(experiment="test2")) == {
                "test2/a-3_b-4"
            }

            result = trial.run()

            assert result == {"result": (1, 2)}

            assert trial.get_result("result") == (1, 2)

            # Write trial data back to the store
            trial.save()

            assert trial_store["test/a-1_b-2"].data["result"] == {"result": (1, 2)}

            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 1}}, experiment)

            experiment.parameter_generator.generators[0].grid["c"] = [1, 2, 3]

            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)
            trial_store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)
