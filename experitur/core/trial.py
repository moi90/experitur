import collections.abc
import copy
import datetime
import glob
import inspect
import itertools
import os.path
import shutil
import traceback
import warnings
from abc import abstractmethod
from typing import Callable, Dict, overload

import yaml

from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
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
    """Decide whether parameters_1 are a subset of parameters_2."""

    # TODO: Ignore _underscore values as these are dynamically calculated

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


def _format_independent_parameters(trial_parameters, independent_parameters):
    if len(independent_parameters) > 0:
        trial_id = "_".join(
            "{}-{!s}".format(k, trial_parameters[k]) for k in independent_parameters
        )
        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return trial_id


class TrialParameters(collections.abc.MutableMapping):
    """
    Parameter configuration of the current trial.

    This is automatically instanciated by experitur and provided to the experiment function:

    .. code-block:: python

        @Experiment(parameters={"a": [1,2,3], "prefix_a": [10]})
        def exp1(parameters):
            # Access current value of parameter `a` (item access)
            parameters["a"]

            # Access extra data (attribute access)
            parameters.id # Trial ID
            parameters.wdir # Trial working directory

            def func(a=1, b=2):
                ...

            # Record default parameters of `func`
            parameters.record_defaults(func)

            # Call `func` with current value of parameter `a` and `b`=5
            parameters.call(func, b=5)

            # Access only parameters starting with a certain prefix
            parameters_prefix = parameters.prefix("prefix_")

            # All the above works as expected:
            # parameters_prefix.<attr>, parameters_prefix[<key>], parameters_prefix.record_defaults, parameters_prefix.call, ...
            # In our case, parameters_prefix["a"] == 10.
    """

    def __init__(self, trial: "Trial", prefix: str = ""):
        self._trial = trial
        self._prefix = prefix

    # MutableMapping provides concrete generic implementations of all
    # methods except for __getitem__, __setitem__, __delitem__,
    # __iter__, and __len__.

    def __getitem__(self, name):
        """Get the value of a parameter."""
        return self._trial.data["resolved_parameters"][f"{self._prefix}{name}"]

    def __setitem__(self, name, value):
        """Set the value of a parameter."""
        self._trial.data["resolved_parameters"][f"{self._prefix}{name}"] = value

    def __delitem__(self, name):
        """Delete a parameter."""
        del self._trial.data["resolved_parameters"][f"{self._prefix}{name}"]

    def __iter__(self):
        start = len(self._prefix)
        return (
            k[start:]
            for k in self._trial.data["resolved_parameters"]
            if k.startswith(self._prefix)
        )

    def __len__(self):
        return sum(1 for k in self)

    def __getattr__(self, name):
        """
        Magic attributes.

        `name` can be one of the following:

        - A trial property:

            - :code:`trial.id`: Trial ID
            - :code:`trial.wdir`: Trial working directory

        - An experiment (:code:`trial.<experiment_name>`):

            This way you can access data of the trial of a
            different experiment with its parameters matching the parameters
            of the current trial.
        """

        # Name could be a data item (e.g. wdir, id, ...)
        try:
            return self._trial.data[name]
        except KeyError:
            pass

        # Name could be a referenced experiment with matching parameters
        trials = self._trial.store.match(
            experiment=name, resolved_parameters=dict(self)
        )

        if len(trials) == 1:
            _, trial = trials.popitem()
            return TrialParameters(trial)
        elif len(trials) > 1:
            msg = "Multiple matching parent experiments: " + ", ".join(trials.keys())
            raise ValueError(msg)

        msg = "Trial has no attribute: {}".format(name)
        raise AttributeError(msg)

    @overload
    def record_defaults(self, func: Callable, **defaults):  # pragma: no cover
        ...

    @overload
    def record_defaults(self, **defaults):  # pragma: no cover
        ...

    def record_defaults(self, *args, **defaults):
        """Record default parameters from a function and/or additional parameters."""

        if len(args) > 1:
            raise ValueError("Only 1 or 2 positional arguments allowed.")

        # First set explicit defaults
        for name, value in defaults.items():
            self.setdefault(name, value)

        if args and callable(args[0]):
            callable_ = args[0]
            for param in inspect.signature(callable_).parameters.values():
                if param.default is not param.empty:
                    self.setdefault(param.name, param.default)

    def call(self, func: Callable, *args, **kwargs):
        """
        Call the function applying the configured parameters.

        Args:
            func (callable): Callable to be applied.
            *args: Positional arguments to the callable.
            **kwargs: Named defaults for the callable.

        Returns:
            The return value of the callable.

        The default values of the callable are determined using ``inspect``.
        Additional defaults can be given using ``**kwargs``.
        These defaults are recorded into the trial.

        As all passed values are recorded, make sure that these have simple
        YAML-serializable types.
        """

        # TODO: partial for complex non-recorded arguments?

        # Record defaults
        self.record_defaults(func, **kwargs)

        # Apply
        callable_names = set(
            param.name
            for param in inspect.signature(func).parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        )

        parameters = {k: v for k, v in self.items() if k in callable_names}

        return func(*args, **parameters)

    def prefixed(self, prefix: str) -> "TrialParameters":
        """
        Return new :py:class:`TrialParameters` instance with prefix applied.

        Prefixes allow you to organize parameters and save keystrokes.
        """
        return TrialParameters(self._trial, f"{self._prefix}{prefix}")


class Trial:
    """
    Trial.

    Life-cycle of a Trial instance
    ------------------------------

    1. Created by sampler
    2. Assigned an ID by TrialStore

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
        """Run the current trial and save the results."""

        # Record intital state
        self.data["success"] = False
        self.data["time_start"] = datetime.datetime.now()
        self.data["result"] = None

        try:
            self.data["result"] = self.callable(TrialParameters(self))
        except (Exception, KeyboardInterrupt) as exc:
            # Log complete exc to file
            with open(os.path.join(self.data["wdir"], "error.txt"), "w") as f:
                f.write(str(exc))
                f.write(traceback.format_exc())

            self.data["error"] = ": ".join(
                filter(None, (exc.__class__.__name__, str(exc)))
            )

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

    @property
    def is_failed(self):
        return not self.data.get("success", False)


class TrialStore(collections.abc.MutableMapping):
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def match(
        self, callable=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> Dict[str, Trial]:
        callable = _callable_to_name(callable)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            experiment = experiment.name

        result = {}
        for trial_id, trial in self.items():
            if (
                callable is not None
                and _callable_to_name(trial.data.get("callable")) != callable
            ):
                continue

            if parameters is not None and not _match_parameters(
                parameters, trial.data.get("parameters", {})
            ):
                continue

            if resolved_parameters is not None and not _match_parameters(
                resolved_parameters, trial.data.get("resolved_parameters", {})
            ):
                continue

            if experiment is not None and trial.data.get("experiment") != str(
                experiment
            ):
                continue

            result[trial_id] = trial

        return result

    def _make_unique_trial_id(
        self, experiment_name, trial_parameters, independent_parameters
    ):
        trial_id = _format_independent_parameters(
            trial_parameters, independent_parameters
        )

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
            return self._make_unique_trial_id(
                experiment_name, trial_parameters, independent_parameters
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

    def create(self, trial_configuration, experiment):
        trial_configuration.setdefault("parameters", {})

        # Calculate trial_id
        trial_id = self._make_unique_trial_id(
            experiment.name,
            trial_configuration["parameters"],
            experiment.independent_parameters,
        )

        wdir = self._make_wdir(trial_id)

        # TODO: Structured experiment meta-data
        trial_configuration = merge_dicts(
            trial_configuration,
            id=trial_id,
            resolved_parameters=RecursiveDict(
                trial_configuration["parameters"], allow_missing=True
            ).as_dict(),
            experiment=experiment.name,
            parent_experiment=experiment.parent.name
            if experiment.parent is not None
            else None,
            result=None,
            callable=_callable_to_name(experiment.callable),
            wdir=wdir,
            experiment_meta=experiment.meta,
        )

        trial = Trial(self, callable=experiment.callable, data=trial_configuration)

        self[trial_id] = trial

        return trial

    def delete_all(self, keys):
        for k in keys:
            del self[k]


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
