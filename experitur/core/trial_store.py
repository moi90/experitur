import collections.abc
import glob
import itertools
import os.path
import shutil
import threading
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Mapping

import yaml

from experitur.core.trial import TrialCollection, TrialData
from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict
from experitur.util import callable_to_name

if TYPE_CHECKING:
    from experitur.core.context import Context


def _match_parameters(parameters_1, parameters_2):
    """Decide whether parameters_1 are a subset of parameters_2."""

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


def _format_independent_parameters(
    trial_parameters: Mapping, independent_parameters: List[str]
):
    if len(independent_parameters) > 0:
        trial_id = "_".join(
            "{}-{!s}".format(k, trial_parameters.get(k, "na"))
            for k in independent_parameters
        )
        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return trial_id


class TrialStore(collections.abc.MutableMapping):
    _implementations: Dict[str, "TrialStore"] = {}

    def __init__(self, ctx: "Context"):
        self.ctx = ctx

    # Class methods for the registration of trial store implementations.
    @classmethod
    def _register_implementation(cls):
        TrialStore._implementations[cls.__name__] = cls

    @staticmethod
    def get_implementation(implementation_name) -> "TrialStore":
        return TrialStore._implementations[implementation_name]

    # Context manager logic: Open and close trial store.
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def open(self):
        """
        Open the trial store.
        
        Overrided in subclasses.
        """

    def close(self):
        """
        Close the trial store.
        
        Overrided in subclasses.
        """

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> TrialCollection:
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            experiment = experiment.name

        trials = []
        for trial in self.values():
            experiment_ = trial.data.get("experiment", {})
            if func is not None and callable_to_name(experiment_.get("func")) != func:
                continue

            if parameters is not None and not _match_parameters(
                parameters, trial.data.get("parameters", {})
            ):
                continue

            if resolved_parameters is not None and not _match_parameters(
                resolved_parameters, trial.data.get("resolved_parameters", {})
            ):
                continue

            if experiment is not None and experiment_.get("name") != str(experiment):
                continue

            trials.append(trial)

        return TrialCollection(trials)

    def _make_unique_trial_id(
        self,
        experiment_name: str,
        trial_parameters: Mapping,
        varying_parameters: List[str],
    ):
        trial_id = _format_independent_parameters(trial_parameters, varying_parameters)

        trial_id = "{}/{}".format(experiment_name, trial_id)

        try:
            existing_trial = self[trial_id]
        except KeyError:
            # If there is no existing trial with this id, it is unique
            return trial_id

        # Otherwise, we have to incorporate more independent parameters
        new_independent_parameters = []

        existing_trial.data.setdefault("parameters", {})

        # Look for parameters in existing_trial that have differing values
        for name, value in existing_trial.data["parameters"].items():
            if name in trial_parameters and trial_parameters[name] != value:
                new_independent_parameters.append(name)

        # Look for parameters that did not exist previously
        for name in trial_parameters.keys():
            if name not in existing_trial.data["parameters"]:
                new_independent_parameters.append(name)

        if new_independent_parameters:
            # If we found parameters where this trial is different from the existing one, append these to independent
            varying_parameters.extend(new_independent_parameters)
            return self._make_unique_trial_id(
                experiment_name, trial_parameters, varying_parameters
            )

        # Otherwise, we just append a version number
        for i in itertools.count(1):
            test_trial_id = "{}.{}".format(trial_id, i)

            try:
                existing_trial = self[test_trial_id]
            except KeyError:
                # If there is no existing trial with this id, it is unique
                return test_trial_id

    def _make_wdir(self, trial_id):
        wdir = os.path.join(self.ctx.wdir, os.path.normpath(trial_id))
        os.makedirs(wdir, exist_ok=True)
        return wdir

    def create(self, trial_configuration, experiment: "Experiment"):
        """Create a :py:class:`TrialData` instance."""
        trial_configuration.setdefault("parameters", {})

        # Calculate trial_id
        trial_id = self._make_unique_trial_id(
            experiment.name,
            trial_configuration["parameters"],
            experiment.varying_parameters,
        )

        wdir = self._make_wdir(trial_id)

        # TODO: Structured experiment meta-data
        trial_configuration = merge_dicts(
            trial_configuration,
            id=trial_id,
            resolved_parameters=RecursiveDict(
                trial_configuration["parameters"], allow_missing=True
            ).as_dict(),
            experiment={
                "name": experiment.name,
                "parent": experiment.parent.name
                if experiment.parent is not None
                else None,
                "func": callable_to_name(experiment.func),
                "meta": experiment.meta,
                # Parameters that where actually configured.
                "independent_parameters": experiment.independent_parameters,
            },
            result=None,
            wdir=wdir,
        )

        trial = TrialData(self, func=experiment.func, data=trial_configuration)

        self[trial_id] = trial

        return trial

    def delete_all(self, keys):
        for k in keys:
            del self[k]


class MemoryTrialStore(TrialStore):
    def __init__(self, ctx: "Context"):
        super().__init__(ctx)
        self.data = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        trial_data = self.data[key]
        return TrialData(self, data=trial_data)

    def __iter__(self):
        return iter(self.data)

    def __setitem__(self, key, trial_data: TrialData):
        self.data[key] = trial_data.data

    def __delitem__(self, key):
        del self.data[key]

    def lock(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()


MemoryTrialStore._register_implementation()  # pylint: disable=protected-access


class FileTrialStore(TrialStore):
    PATTERN = os.path.join("{}", "trial.yaml")
    DUMPER = ExperiturDumper

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, key):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format(key))
        path = os.path.normpath(path)

        try:
            with open(path) as fp:
                return TrialData(self, data=yaml.load(fp, Loader=yaml.Loader))
        except FileNotFoundError as exc:
            raise KeyError from exc

    def __iter__(self):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format("**"))
        path = os.path.normpath(path)

        left, right = path.split("**", 1)

        for entry_fn in glob.iglob(path, recursive=True):
            if os.path.isdir(entry_fn):
                continue

            # Convert entry_fn back to key
            k = entry_fn[len(left) : -len(right)]

            # Keys use forward slashes
            k = k.replace("\\", "/")

            yield k

    def __setitem__(self, key, trial_data: TrialData):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format(key))
        path = os.path.normpath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as fp:
            yaml.dump(trial_data.data, fp, Dumper=self.DUMPER)

        # raise KeyError

    def __delitem__(self, key):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format(key))

        try:
            os.remove(path)
        except FileNotFoundError:
            raise KeyError

        shutil.rmtree(os.path.dirname(path))

    def lock(self):
        # TODO: Implement locking
        pass

    def release(self):
        # TODO: Implement locking
        pass


# Make implementation known to the base class
FileTrialStore._register_implementation()  # pylint: disable=protected-access

try:
    import zerorpc
except ImportError:
    pass
else:

    class RemoteFileTrialStore(TrialStore):
        def __init__(self, ctx: "Context"):
            super().__init__(ctx)

            endpoint = self.ctx.config["remote_endpoint"]
            self.client = zerorpc.Client(endpoint)

        def create(self, trial_configuration, experiment: "Experiment"):
            return self.client.create_trial(trial_id)

        def __delitem__(self, trial_id):
            self.client.del_trial(trial_id)

        def __getitem__(self, trial_id) -> TrialData:
            return TrialData(self, self.client.get_trial_data(trial_id))

        def __setitem__(self, trial_id, trial_data: TrialData):
            return self.client.set_trial_data(trial_id, trial_data.data)

        def __iter__(self):
            return iter(self.client.get_trial_ids())

        def __len__(self):
            return self.client.get_n_trials()

    # RemoteFileTrialStore._register_implementation()  # pylint: disable=protected-access

