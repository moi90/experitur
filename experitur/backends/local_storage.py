import errno
import os
from abc import abstractmethod

import yaml

from .base import BaseBackend


class LocalStorageBackend(BaseBackend):
    def __init__(self):
        self._trials = {}

    @abstractmethod
    def reload(self):
        raise NotImplementedError

    def find_trials_by_parameters(self, parameters):
        result = {}

        for trial_id, trial_data in self._trials.items():
            trial_parameters = trial_data["parameters_post"]

            if self._is_match(parameters, trial_parameters):
                result[trial_id] = trial_data

        return result

    def get_trial_by_id(self, trial_id):
        return self._trials[trial_id]

    def trials(self):
        return self._trials.items()
