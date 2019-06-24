import collections.abc
import datetime
import glob
import inspect
import itertools
import os.path
import shutil
import traceback
import warnings
from abc import abstractmethod

import yaml

from experitur.helpers.dumper import ExperiturDumper
from experitur.recursive_formatter import RecursiveDict


def _callable_to_name(obj):
    if callable(obj):
        return "{}.{}".format(obj.__module__, obj.__name__)

    if isinstance(obj, list):
        return [_callable_to_name(x) for x in obj]

    if isinstance(obj, dict):
        return {_callable_to_name(k): _callable_to_name(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(_callable_to_name(x) for x in obj)

    return obj


def _match_parameters(parameters_1, parameters_2):
    """
    Decide whether parameters_1 are a subset of parameters_2.
    """

    # TODO: Ignore _underscore values as these are dynamically calculated

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


def _format_independent_parameters(trial_parameters, independent_parameters):
    if len(independent_parameters) > 0:
        trial_id = "_".join("{}-{!s}".format(k, trial_parameters[k])
                            for k in independent_parameters)
        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return trial_id


class TrialProxy(collections.abc.MutableMapping):
    """
    This is the trial object that the experiment interacts with.
    """

    def __init__(self, trial):
        self._trial = trial
        self._parameters = RecursiveDict(
            self._trial.data["parameters"], allow_missing=True)

    def __getitem__(self, name):
        return self._parameters[name]

    def __setitem__(self, name, value):
        self._trial.data["parameters"][name] = value

    def __delitem__(self, name):
        del self._trial.data["parameters"][name]

    def __iter__(self):
        return iter(self._trial.data["parameters"])

    def __len__(self):
        return len(self._trial.data["parameters"])

    def __getattr__(self, name):
        """
        Magic attributes.
        """

        # Name could be a data item (e.g. wdir, id, ...)
        try:
            return self._trial.data[name]
        except KeyError:
            pass

        # Name could be a referenced experiment with matching parameters
        trials = self._trial.store.match(
            experiment=name, parameters=self.parameters)

        if len(trials) == 1:
            _, trial = trials.popitem()
            return TrialProxy(trial)
        elif len(trials) > 1:
            msg = "Multiple matching parent experiments: " + \
                ", ".join(trials.keys())
            raise ValueError(msg)

        msg = "Trial has no attribute: {}".format(name)
        raise AttributeError(msg)

    def record_defaults(self, prefix, *args, **defaults):
        """
        Set default parameters.

        Default parameters can be assigned directly or guessed from a callable.
        """
        if len(args) > 1:
            raise ValueError("Only 1 or 2 positional arguments allowed.")

        # First set explicit defaults
        for name, value in defaults.items():
            self.setdefault(prefix + name, value)

        if args and callable(args[0]):
            callable_ = args[0]
            for param in inspect.signature(callable_).parameters.values():
                if param.default is not param.empty:
                    self.setdefault(prefix + param.name, param.default)

    def apply(self, prefix, callable_, *args, **kwargs):
        callable_names = set(
            param.name
            for param in inspect.signature(callable_).parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD)

        start = len(prefix)
        parameters = {
            k[start:]: v
            for k, v in self.items()
            if k.startswith(prefix) and k[start:] in callable_names
        }

        intersection_names = set(kwargs.keys()) & set(parameters.keys())
        if intersection_names:
            warnings.warn("Redefining parameter(s) {} with keyword parameter.".format(
                ", ".join(intersection_names)))

        for k, v in kwargs.items():
            parameters[k] = v

        return callable_(*args, **parameters)

    def without_prefix(self, prefix):
        """
        Extract parameters beginning with prefix and remove the prefix.
        """
        start = len(prefix)

        return {
            k[start:]: v
            for k, v in self.items()
            if k.startswith(prefix)
        }


class Trial:
    """
    Arguments
        store: TrialStore
        callable (optional): Experiment callable
        data (optional): Trial data
    """

    def __init__(self, store, callable=None, data=None):
        self.store = store
        self.callable = callable
        self.data = data or {}

    def run(self):
        """
        Run the current trial and save the results.
        """

        # Record intital state
        self.data["success"] = False
        self.data["time_start"] = datetime.datetime.now()
        self.data["result"] = None

        try:
            self.data["result"] = self.callable(TrialProxy(self))
        except (Exception, KeyboardInterrupt) as exc:
            # Log complete exc to file
            with open(os.path.join(self.data["wdir"], "error.txt"), "w") as f:
                f.write(str(exc))
                f.write(traceback.format_exc())

            self.data["error"] = ": ".join(
                filter(None, (exc.__class__.__name__, str(exc))))

            raise exc

        else:
            self.data["success"] = True
        finally:
            self.data["time_end"] = datetime.datetime.now()
            self.save()

        return self.data["result"]

    def save(self):
        self.store[self.data["id"]] = self

    @property
    def id(self):
        return self.data["id"]


class TrialStore(collections.abc.MutableMapping):
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def match(self, callable=None, parameters=None, experiment=None):
        callable = _callable_to_name(callable)

        result = {}
        for trial_id, trial in self.items():
            if callable is not None and _callable_to_name(trial.data.get("callable")) != callable:
                continue

            if parameters is not None and not _match_parameters(parameters, trial.data.get("parameters", {})):
                continue

            if experiment is not None and trial.data.get("experiment") != experiment:
                continue

            result[trial_id] = trial

        return result

    def _make_unique_trial_id(self, experiment_name, trial_parameters, independent_parameters):
        trial_id = _format_independent_parameters(
            trial_parameters, independent_parameters)

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
            independent_parameters.extend(new_independent_parameters)
            return self._make_unique_trial_id(experiment_name, trial_parameters, independent_parameters)

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

    def create(self, parameters, experiment):
        # Calculate trial_id
        trial_id = self._make_unique_trial_id(
            experiment.name,
            parameters,
            experiment.independent_parameters)

        wdir = self._make_wdir(trial_id)

        trial_data = {
            "id": trial_id,
            "experiment": experiment.name,
            "parent_experiment": experiment.parent.name if experiment.parent is not None else None,
            "result": None,
            "parameters": parameters,
            "callable": _callable_to_name(experiment.callable),
            "wdir": wdir,
        }

        trial = Trial(self, callable=experiment.callable, data=trial_data)

        self[trial_id] = trial

        return trial


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
                return Trial(self, data=yaml.load(fp, Loader=yaml.Loader))
        except FileNotFoundError:
            raise KeyError

    def __iter__(self):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format("**"))
        path = os.path.normpath(path)

        left, right = path.split("**", 1)

        for entry_fn in glob.iglob(path, recursive=True):
            if os.path.isdir(entry_fn):
                continue

            # Convert entry_fn back to key
            k = entry_fn[len(left):-len(right)]

            # Keys use forward slashes
            k = k.replace("\\", "/")

            yield k

    def __setitem__(self, key, value):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format(key))
        path = os.path.normpath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as fp:
            yaml.dump(value.data, fp, Dumper=self.DUMPER)

        # raise KeyError

    def __delitem__(self, key):
        path = os.path.join(self.ctx.wdir, self.PATTERN.format(key))

        try:
            os.remove(path)
        except FileNotFoundError:
            raise KeyError

        shutil.rmtree(os.path.dirname(path))
