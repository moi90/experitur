from experitur.core.context import Context
from experitur.core.root_trial_collection import _format_trial_id


def test__format_trial_id():
    parameters = {"a": 1, "b": 2}
    assert _format_trial_id("foo", parameters, []) == "foo/_"
    assert _format_trial_id("foo", parameters, ["a", "b"]) == "foo/a-1_b-2"


def test_RootTrialCollection(tmp_path):
    with Context(str(tmp_path), writable=True) as ctx:
        ctx: Context

        # Check that creating the same trial a second time appends a number
        ctx.trials.create(
            {
                "experiment": {"name": "test2", "varying_parameters": []},
            }
        )

        trial = ctx.trials.create(
            {
                1: "foo",
                "bar": 2,
                "experiment": {"name": "test2", "varying_parameters": []},
            }
        )

        trial_wdir = ctx.get_trial_wdir("test2/_.1")

        assert trial.id == "test2/_.1"
        del trial._data["revision"]
        assert trial._data == {  # pytest: disable=protected-access
            "id": "test2/_.1",
            1: "foo",
            "bar": 2,
            "parameters": {},
            "resolved_parameters": {},
            "experiment": {"name": "test2", "varying_parameters": []},
            "wdir": trial_wdir,
            "used_parameters": [],
        }

        trial = ctx.trials.create(
            {
                1: "foo",
                "bar": 2,
                "experiment": {"name": "test2", "varying_parameters": []},
                "parameters": {"a": 1},
            },
        )
        assert trial.id == "test2/a-1"
