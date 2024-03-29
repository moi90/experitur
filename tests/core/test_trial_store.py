import os.path
from typing import Type

import pytest

from experitur.core.context import Context
from experitur.core.experiment import Experiment
from experitur.core.trial_store import (
    FileTrialStore,
    KeyExistsError,
    TrialStore,
    _match_parameters,
)
from experitur.helpers.merge_dicts import merge_dicts


def test__match_parameters():
    assert _match_parameters({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 4, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _match_parameters({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})


@pytest.fixture(name="TrialStoreImplementation", params=[FileTrialStore])
def _TrialStoreImplementation(request):
    return request.param


def test_trial_store(tmp_path, TrialStoreImplementation: Type[TrialStore]):
    with Context(str(tmp_path), writable=True) as ctx:
        ctx: Context
        trial_store: TrialStore = TrialStoreImplementation(ctx)

        @Experiment("test", configurator={"a": [1, 2], "b": [2, 3]})
        def test(_):
            return {"result": (1, 2)}

        @Experiment("test2", parent=test)
        def test2(_):  # pylint: disable=unused-variable
            return {"result": (2, 4)}

        trial_data = trial_store.create(
            "test2/_", {"experiment": {"name": "test2", "varying_parameters": []}}
        )
        assert "id" in trial_data
        assert trial_data["id"] == "test2/_"

        result = trial_store["test2/_"]
        del result["revision"]
        assert result == {
            "id": "test2/_",
            "experiment": {"name": "test2", "varying_parameters": []},
        }

        trial_store["test2/_"] = merge_dicts(
            trial_store["test2/_"], foo="bar", bar="baz"
        )
        result = trial_store["test2/_"]
        del result["revision"]
        assert result == {
            "id": "test2/_",
            "foo": "bar",
            "bar": "baz",
            "experiment": {"name": "test2", "varying_parameters": []},
        }

        # Check that storing an unknown trial_id raises a KeyError
        with pytest.raises(KeyError):
            trial_store["foo"] = {}

        # Check that accessing an unknown trial_id raises a KeyError
        with pytest.raises(KeyError):
            _ = trial_store["foo"]

        # Check that deleting an unknown trial_id raises a KeyError
        with pytest.raises(KeyError):
            trial_store.delete("foo")

        # Check that creating the same key twices raises a  KeyExistsError
        trial_store.create("existing", {})
        with pytest.raises(KeyExistsError):
            trial_store.create("existing", {})

        trial_store.delete("existing")

        trial_data = trial_store.create(
            "test3/a-1_b-2",
            {
                "parameters": {"a": 1, "b": 2},
                "experiment": {"name": "test3", "varying_parameters": ["a", "b"]},
            },
        )
        assert trial_data["id"] == "test3/a-1_b-2"
        assert trial_store["test3/a-1_b-2"] == trial_data
        assert "test3/a-1_b-2" in trial_store

        # Check that a fake folder does not disturb the store
        fake_folder = os.path.join(ctx.wdir, "fake", trial_store.TRIAL_FN)
        os.makedirs(fake_folder, exist_ok=True)
        assert "fake" not in trial_store

        # Check that match(func) works
        trial_data = trial_store.create(
            "func_test1/_",
            {
                "experiment": {
                    "name": "func_test1",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            },
        )
        trial_data = trial_store.create(
            "func_test2/_",
            {
                "experiment": {
                    "name": "func_test2",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            },
        )
        trial_data = trial_store.create(
            "func_test3/_",
            {
                "experiment": {
                    "name": "func_test3",
                    "varying_parameters": [],
                    "func": "foo",
                },
                "parameters": {"a": 1, "b": 2},
            },
        )

        result = trial_store.match(func="foo")

        assert isinstance(result, list)
        assert all(isinstance(td, dict) for td in result)

        assert set(trial_data["id"] for trial_data in result) == {
            "func_test1/_",
            "func_test2/_",
            "func_test3/_",
        }

        # parameters
        assert set(
            trial["id"] for trial in trial_store.match(parameters={"a": 1, "b": 2})
        ) == {"func_test1/_", "func_test2/_", "func_test3/_", "test3/a-1_b-2",}

        # experiment
        assert set(
            trial_data["id"]
            for trial_data in trial_store.match(experiment="func_test1")
        ) == {"func_test1/_"}

        # Set context read-only
        ctx.writable = False
        with pytest.raises(RuntimeError):
            trial_store["foo"] = None

        with pytest.raises(RuntimeError):
            trial_store.delete("foo")

        with pytest.raises(RuntimeError):
            trial_store.delete_all([])


def test_trial_store_write(
    tmp_path, TrialStoreImplementation: Type[TrialStore], benchmark
):
    with Context(str(tmp_path), writable=True) as ctx:
        ctx: Context
        trial_store: TrialStore = TrialStoreImplementation(ctx)

        @Experiment("test", configurator={"a": [1, 2], "b": [2, 3]})
        def test(_):
            return {"result": (1, 2)}

        @Experiment("test2", parent=test)
        def test2(_):  # pylint: disable=unused-variable
            return {"result": (2, 4)}

        trial_data = trial_store.create(
            "test2/_", {"experiment": {"name": "test2", "varying_parameters": []},}
        )
        assert "id" in trial_data
        assert trial_data["id"] == "test2/_"

        trial_data = merge_dicts(trial_data, foo="bar", bar="baz")

        # Benchmark writing
        benchmark(trial_store.__setitem__, "test2/_", trial_data)

        trial_data = trial_store["test2/_"]
        del trial_data["revision"]

        assert trial_data == {
            "id": "test2/_",
            "foo": "bar",
            "bar": "baz",
            "experiment": {"name": "test2", "varying_parameters": []},
        }


def test_trial_store_read(
    tmp_path, TrialStoreImplementation: Type[TrialStore], benchmark
):
    with Context(str(tmp_path), writable=True) as ctx:
        ctx: Context
        trial_store: TrialStore = TrialStoreImplementation(ctx)

        trial_data = trial_store.create(
            "test2/_", {"experiment": {"name": "test2", "varying_parameters": []},}
        )

        assert "id" in trial_data
        assert trial_data["id"] == "test2/_"

        # Benchmark reading
        result = benchmark(trial_store.__getitem__, "test2/_")
        del result["revision"]
        assert result == {
            "id": "test2/_",
            "experiment": {"name": "test2", "varying_parameters": [],},
        }

