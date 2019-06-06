from experitur.backends.base import BaseBackend
import errno
import os
from abc import abstractmethod

import yaml


class LocalStorageBackend(BaseBackend):
    def __init__(self):
        self._trials = {}

    @abstractmethod
    def reload(self):
        raise NotImplementedError

    def find_trials_by_parameters(self, clbl, parameters):
        if callable(clbl):
            clbl = clbl.__name__

        if not isinstance(clbl, str):
            raise ValueError("type(clbl) should be callable or str.")

        result = {}

        for trial_id, trial_data in self._trials.items():
            print(trial_data)
            if trial_data.get("callable") != clbl:
                continue

            if not self._is_match(parameters, trial_data["parameters_post"]):
                continue

            result[trial_id] = trial_data

        return result

    def get_trial_by_id(self, trial_id):
        return self._trials[trial_id]

    def trials(self):
        return self._trials.items()

    def add_trial(self, trial_dict):
        if trial_dict.id in self._trials:
            raise Exception(
                "A trial with this ID already exists: {}".format(trial_dict.id))
