import os.path
from typing import Type

import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial_store import (
    FileTrialStore,
    TrialStore,
    _format_trial_id,
    _match_parameters,
)
from experitur.helpers.merge_dicts import merge_dicts


def test__match_parameters():
    assert _match_parameters({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 4, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})


def test__format_independent_parameters():
    parameters = {"a": 1, "b": 2}
    assert _format_trial_id("foo", parameters, []) == "foo/_"
    assert _format_trial_id("foo", parameters, ["a", "b"]) == "foo/a-1_b-2"


@pytest.fixture(name="TrialStoreImplementation", params=[FileTrialStore])
def _TrialStoreImplementation(request):
    return request.param


def test_trial_store(tmp_path, TrialStoreImplementation: Type[TrialStore]):
    with Context(str(tmp_path), writable=True) as ctx:
        ctx: Context
        trial_store: TrialStore = TrialStoreImplementation(ctx)

        @Experiment("test", parameters={"a": [1, 2], "b": [2, 3]})
        def test(_):
            return {"result": (1, 2)}

        @Experiment("test2", parent=test)
        def test2(_):  # pylint: disable=unused-variable
            return {"result": (2, 4)}

        trial_data = trial_store.create(
            {"experiment": {"name": "test2", "varying_parameters": []}}
        )
        assert "id" in trial_data
        assert trial_data["id"] == "test2/_"

        assert trial_store["test2/_"] == {
            "id": "test2/_",
            "parameters": {},
            "resolved_parameters": {},
            "experiment": {"name": "test2", "varying_parameters": []},
        }

        trial_store["test2/_"] = merge_dicts(
            trial_store["test2/_"], foo="bar", bar="baz"
        )
        assert trial_store["test2/_"] == {
            "id": "test2/_",
            "parameters": {},
            "foo": "bar",
            "bar": "baz",
            "resolved_parameters": {},
            "experiment": {"name": "test2", "varying_parameters": []},
        }

        # Check that storing an unknown trial_id raises a KeyError
        with pytest.raises(KeyError):
            trial_store["foo"] = {}

        # Check that accessing an unknown trial_id raises a KeyError
        with pytest.raises(KeyError):
            _ = trial_store["foo"]

        trial_data = trial_store.create(
            {
                1: "foo",
                "bar": 2,
                "experiment": {"name": "test2", "varying_parameters": []},
            }
        )
        assert "id" in trial_data
        assert trial_data["id"] == "test2/_.1"
        assert trial_store["test2/_.1"] == {
            "id": "test2/_.1",
            1: "foo",
            "bar": 2,
            "parameters": {},
            "resolved_parameters": {},
            "experiment": {"name": "test2", "varying_parameters": []},
        }

        trial_data = trial_store.create(
            {
                1: "foo",
                "bar": 2,
                "experiment": {"name": "test2", "varying_parameters": []},
                "parameters": {"a": 1},
            }
        )
        assert trial_data["id"] == "test2/a-1"

        trial = trial_store.create(
            {
                "parameters": {"a": 1, "b": 2},
                "experiment": {"name": "test3", "varying_parameters": ["a", "b"]},
            }
        )

        fake_folder = os.path.join(ctx.wdir, "fake", trial_store.TRIAL_FN)
        os.makedirs(fake_folder, exist_ok=True)

        assert trial["id"] == "test3/a-1_b-2"

        assert "test3/a-1_b-2" in trial_store

        assert len(trial_store) == 4

        del trial_store["test2/_"]
        del trial_store["test2/_.1"]

        assert len(trial_store) == 2

        with pytest.raises(KeyError):
            del trial_store["foo"]

        # Check that match(func) works
        trial_data = trial_store.create(
            {
                "experiment": {
                    "name": "func_test1",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            }
        )
        trial_data = trial_store.create(
            {
                "experiment": {
                    "name": "func_test2",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            }
        )
        trial_data = trial_store.create(
            {
                "experiment": {
                    "name": "func_test3",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            }
        )

        result = trial_store.match(func="foo")

        assert isinstance(result, list)
        assert all(isinstance(td, dict) for td in result)

        assert set(trial["id"] for trial in result) == {
            "func_test1/_",
            "func_test2/_",
            "func_test3/_",
        }

        assert set(
            trial["id"] for trial in trial_store.match(parameters={"a": 1, "b": 2})
        ) == {
            "func_test1/_",
            "func_test2/_",
            "func_test3/_",
            "test3/a-1_b-2",
        }

        # Set context read-only
        ctx.writable = False
        with pytest.raises(RuntimeError):
            trial_store["foo"] = None

        with pytest.raises(RuntimeError):
            del trial_store["foo"]

        with pytest.raises(RuntimeError):
            trial_store.delete_all([])
