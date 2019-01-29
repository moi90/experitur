from experitur.backends.local_storage import LocalStorageBackend
from copy import deepcopy


class MockBackend(LocalStorageBackend):
    def reload(self):
        self._trials = {

        }

    def set_trial(self, trial_id, trial_data):
        self._trials[trial_id] = deepcopy(trial_data)


def test_local_storage():
    backend = MockBackend()
    backend.reload()

    parameters = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    # No independent parameters
    trial_id = backend.make_trial_id(parameters, [])

    assert trial_id == "_"
    assert trial_id not in backend._trials

    backend.set_trial(trial_id, {"parameters_post": parameters})

    assert trial_id in backend._trials

    assert list(backend.trials()) == [("_", {"parameters_post": parameters})]

    # No matching parameter set (parameter exists)
    result = backend.find_trials_by_parameters({"c": 4})
    assert result == {}

    # No matching parameter set (parameter does not exist)
    result = backend.find_trials_by_parameters({"d": 1})
    assert result == {}

    # One matching parameter set
    result = backend.find_trials_by_parameters({"c": 3})
    assert result == {"_": {"parameters_post": parameters}}

    # Vary one parameter
    for i in range(3):
        parameters["c"] = i
        trial_id = backend.make_trial_id(parameters, ["c"])

        assert trial_id not in backend._trials

        backend.set_trial(trial_id, {"parameters_post": parameters})

        assert trial_id in backend._trials

    # Add one parameter
    parameters["d"] = 10

    independent_parameters = []
    trial_id = backend.make_trial_id(parameters, independent_parameters)
    assert independent_parameters == ["c", "d"]
    assert trial_id not in backend._trials

    backend.set_trial(trial_id, {"parameters_post": parameters})

    assert trial_id in backend._trials

    # Same again to get a versioned id
    independent_parameters = []
    trial_id = backend.make_trial_id(parameters, independent_parameters)
    assert independent_parameters == ["c", "d"]
    assert trial_id not in backend._trials
    assert trial_id == "c-2_d-10.0"

    backend.set_trial(trial_id, {"parameters_post": parameters})

    assert trial_id in backend._trials
