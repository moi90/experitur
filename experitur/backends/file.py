import errno
import os

import yaml

from .local_storage import LocalStorageBackend


class FileBackend(LocalStorageBackend):
    def __init__(self, experiment_root):
        super(FileBackend, self).__init__()

        self.experiment_root = experiment_root

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
