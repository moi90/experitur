import errno
import os

import yaml

from .base import BaseBackend


class FileBackend(BaseBackend):
    def __init__(self, experiment_root):
        self.experiment_root = experiment_root
        self._trials = {}

        self.reload()

    def reload(self):
        """
        Reload data from self.path.
        """

        for trial_id in os.listdir(self.experiment_root):
            try:
                with open(os.path.join(self.experiment_root, trial_id, "experitur.yaml")) as fp:
                    self._trials[trial_id] = yaml.load(fp)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    continue
                raise

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
