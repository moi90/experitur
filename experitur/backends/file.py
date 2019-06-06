import errno
import os

import yaml

from experitur.backends.local_storage import LocalStorageBackend
from experitur.helpers.dumper import ExperiturDumper


class FileBackend(LocalStorageBackend):
    def __init__(self, path):
        super(FileBackend, self).__init__()

        self.path = path

        self.reload()

    def reload(self):
        """
        Reload data from self.path.
        """

        print("FileBackend.reload:", self.path)

        for experiment_entry in os.scandir(self.path):
            if not experiment_entry.is_dir():
                continue

            for trial_entry in os.scandir(experiment_entry.path):
                if not trial_entry.is_dir():
                    continue

                try:
                    with open(os.path.join(trial_entry.path, "experitur.yaml")) as fp:
                        trial_dict = yaml.load(fp)
                        self._trials[trial_dict.id] = trial_dict
                except FileNotFoundError:
                    pass

    def save_trial(self, trial):
        trial_dict = trial.get_trial_dict()

        trial_path = os.path.join(self.path, trial.experiment.name, trial.id)
        os.makedirs(trial_path, exist_ok=True)
        with open(os.path.join(trial_path, "experitur.yaml"), "w") as fp:
            yaml.dump(trial_dict, fp, Dumper=ExperiturDumper)
