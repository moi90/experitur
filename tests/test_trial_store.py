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


def test_file_trial_store(tmp_path):
    config = {"store": "FileTrialStore"}
    with Context(str(tmp_path), config) as ctx:
        ctx: Context
        with ctx.store:
            ctx.store: FileTrialStore
            # Check that folders named {}/trial.yaml do not disturb the store
            fake_folder = os.path.join(ctx.wdir, ctx.store.PATTERN.format("fake"))
            os.makedirs(fake_folder, exist_ok=True)
            assert len(ctx.store) == 0


@pytest.mark.parametrize(
    "store", TrialStore._implementations.keys()  # pylint: disable=protected-access
)
def test_trial_store(tmp_path_factory, store, random_ipc_endpoint):
    config = {"store": store}

    if store == "RemoteFileTrialStore":
        config["remote_endpoint"] = random_ipc_endpoint

        #  Start server
        import gevent

        from experitur.server import ExperiturServer

        server_config = {"store": "MemoryTrialStore"}
        server_ctx = Context(str(tmp_path_factory.mktemp("server_ctx")), server_config)
        server = ExperiturServer(server_ctx)
        server.bind(random_ipc_endpoint)
        gevent.spawn(server.run)

    with Context(str(tmp_path_factory.mktemp("client_ctx")), config) as ctx:
        ctx: Context
        with ctx.store:

            def test(trial):
                return {"result": (1, 2)}

            experiment = Experiment("test", parameters={"a": [1, 2], "b": [2, 3]})(test)

            def test2(trial):
                return {"result": (2, 4)}

            experiment2 = Experiment("test2", parent=experiment)(test2)

            ctx.store["foo"] = TrialData(
                ctx.store, data={"id": "foo", "wdir": "", 1: "foo", "bar": 2}
            )
            assert ctx.store["foo"].data == {
                "id": "foo",
                "wdir": "",
                1: "foo",
                "bar": 2,
            }

            ctx.store["bar/baz"] = TrialData(
                ctx.store, data={"id": "bar/baz", "wdir": "", 1: "foo", "bar": 2}
            )
            assert ctx.store["bar/baz"].data == {
                "id": "bar/baz",
                "wdir": "",
                1: "foo",
                "bar": 2,
            }

            trial = ctx.store.create({"parameters": {"a": 1, "b": 2}}, experiment)

            assert trial.data["id"] == "test/a-1_b-2"

            assert "test/a-1_b-2" in ctx.store

            assert len(ctx.store) == 3

            del ctx.store["bar/baz"]
            del ctx.store["foo"]

            with pytest.raises(KeyError):
                del ctx.store["foo"]

            ctx.store.create({"parameters": {"a": 2, "b": 3}}, experiment)
            ctx.store.create({"parameters": {"a": 3, "b": 4}}, experiment)
            ctx.store.create({"parameters": {"a": 3, "b": 4}}, experiment2)

            assert set(trial.id for trial in ctx.store.match(func=test)) == {
                "test/a-1_b-2",
                "test/a-2_b-3",
                "test/a-3_b-4",
            }

            assert set(
                trial.id for trial in ctx.store.match(parameters={"a": 1, "b": 2})
            ) == {"test/a-1_b-2"}

            assert set(trial.id for trial in ctx.store.match(experiment="test2")) == {
                "test2/a-3_b-4"
            }

            result = trial.run()

            assert result == {"result": (1, 2)}

            assert trial.get_result("result") == (1, 2)

            # Write trial data back to the store
            trial.save()

            assert ctx.store["test/a-1_b-2"].data["result"] == {"result": (1, 2)}

            ctx.store.create({"parameters": {"a": 1, "b": 2, "c": 1}}, experiment)

            experiment.parameter_generator.generators[0].grid["c"] = [1, 2, 3]

            ctx.store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)
            ctx.store.create({"parameters": {"a": 1, "b": 2, "c": 2}}, experiment)
