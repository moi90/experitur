import collections.abc
import contextlib
import glob
import os.path
import tempfile
import typing
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional

import yaml
from filelock import SoftFileLock

from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.util import callable_to_name

if typing.TYPE_CHECKING:
    from experitur.core.context import Context


def _match_parameters(parameters_1, parameters_2):
    """Decide whether parameters_1 are a subset of parameters_2."""

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


class KeyExistsError(Exception):
    pass


class TrialStore(collections.abc.MutableMapping):
    def __init__(self, ctx: "Context"):
        self.ctx = ctx

    @abstractmethod
    def iter(self, prefix: Optional[str] = None) -> Iterator[str]:
        pass

    def __iter__(self):
        return self.iter()

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> List[Dict]:
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            if experiment.name is None:
                raise ValueError(f"Experiment {experiment!r} has no name set")
            experiment = experiment.name

        if experiment is not None:
            prefix = experiment + "/"
        else:
            prefix = None

        trial_data_list = []
        for trial_id in self.iter(prefix):
            trial_data = self[trial_id]

            experiment_ = trial_data.get("experiment", {})
            if func is not None and callable_to_name(experiment_.get("func")) != func:
                continue

            if parameters is not None and not _match_parameters(
                parameters, trial_data.get("parameters", {})
            ):
                continue

            if resolved_parameters is not None and not _match_parameters(
                resolved_parameters, trial_data.get("resolved_parameters", {})
            ):
                continue

            if experiment is not None and experiment_.get("name") != experiment:
                continue

            trial_data_list.append(trial_data)

        return trial_data_list

    @abstractmethod
    def _create(self, trial_data):
        """
        Create an entry using the trial_data["id"].

        Returns:
            trial_data

        Raises:
            KeyExistsError if a trial with the specified id already exists.
        """

    def create(self, trial_id, trial_data):
        return self._create(merge_dicts(trial_data, id=trial_id))

    def check_writable(self):
        __tracebackhide__ = True  # pylint: disable=unused-variable

        if not self.ctx.writable:
            raise RuntimeError("Context is not writable.")

    def delete_all(self, keys):
        self.check_writable()

        for k in keys:
            del self[k]


class FileTrialStore(TrialStore):
    TRIAL_FN = "trial.yaml"
    DUMPER = ExperiturDumper

    def __init__(self, ctx: "Context"):
        super().__init__(ctx)

        self._lock = SoftFileLock(
            os.path.join(ctx.wdir, "trial_store.lock"), timeout=10
        )

    def _get_trial_fn(self, trial_id):
        return os.path.join(self.ctx.get_trial_wdir(trial_id), self.TRIAL_FN)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, trial_id):
        path = self._get_trial_fn(trial_id)

        try:
            with open(path) as f:
                trial_data = yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError as exc:
            raise KeyError(trial_id) from exc
        except:
            print(f"Error reading {path}")
            raise

        # Make sure that the trial_id always matches the requested trial_id.
        # The stored trial_id can differ if the directory was moved.
        trial_data["id"] = trial_id
        return trial_data

    def iter(self, prefix=None):

        if prefix is None:
            pattern = "**"
        else:
            pattern = prefix + "**"

        path = self._get_trial_fn(pattern)

        left, right = path.split(pattern, 1)

        with self._lock:
            for entry_fn in glob.iglob(path, recursive=True):
                if os.path.isdir(entry_fn):
                    continue

                if ".trash" in entry_fn.split(os.path.sep):
                    continue

                # Convert entry_fn back to key
                k = entry_fn[len(left) : -len(right)]

                # Keys use forward slashes
                k = k.replace("\\", "/")

                yield k

    def _create(self, trial_data: dict):
        trial_id = trial_data["id"]

        if not isinstance(trial_data, dict):
            raise ValueError(f"trial_data has to be dict, got {trial_data!r}")

        wdir = self.ctx.get_trial_wdir(trial_id)

        with self._lock:
            try:
                os.makedirs(wdir, exist_ok=False)
            except FileExistsError:
                raise KeyExistsError(trial_id) from None

            path = self._get_trial_fn(trial_id)
            with open(path, "x") as fp:
                yaml.dump(trial_data, fp, Dumper=self.DUMPER)

            return trial_data

    def __setitem__(self, trial_id: str, trial_data: dict):
        self.check_writable()

        if not isinstance(trial_data, dict):
            raise ValueError(f"trial_data has to be dict, got {trial_data!r}")

        path = self._get_trial_fn(trial_id)

        # Prevent concurrent modification of the data file
        lock_fn = os.path.dirname(path) + ".lock"
        with SoftFileLock(lock_fn, timeout=10):
            # Check that file exists
            if not os.path.isfile(path):
                raise KeyError(trial_id)

            # Write new contents atomically
            with self._open_atomic(path, "w") as fp:
                yaml.dump(trial_data, fp, Dumper=self.DUMPER)

    @staticmethod
    @contextlib.contextmanager
    def _open_atomic(fn, mode):
        """
        Open a file atomically.
        
        The context manager returns a temporary file handle
        and uses os.replace to replace the actual file with it after it has been written.
        """

        path = os.path.dirname(fn)

        # Create temporary file
        temp_fh = tempfile.NamedTemporaryFile(mode=mode, dir=path, delete=False,)

        yield temp_fh

        # Reliably flush file contents
        temp_fh.flush()
        os.fsync(temp_fh.fileno())
        temp_fh.close()

        # Cleanup
        try:
            os.replace(temp_fh.name, fn)
        finally:
            try:
                os.remove(temp_fh.name)
            except:  # pylint: disable=bare-except
                pass

    def __delitem__(self, trial_id):
        self.check_writable()

        with self._lock:
            old_path = self.ctx.get_trial_wdir(trial_id)

            if not os.path.isdir(old_path):
                raise KeyError(trial_id)

            new_path_base = new_path = os.path.normpath(
                os.path.join(self.ctx.wdir, ".trash", os.path.normpath(trial_id))
            )

            i = 0
            while True:
                if not os.path.isdir(new_path):
                    break

                new_path = f"{new_path_base}.{i:d}"
                i += 1

            os.renames(old_path, new_path)

