import collections.abc
import copy
import datetime
import glob
import inspect
import itertools
import os.path
import pickle
import shutil
import traceback
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Collection
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

import atomicwrites
import yaml

from experitur.core.logger import YAMLLogger
from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment

T = TypeVar("T")


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

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


def _format_independent_parameters(
    trial_parameters: Mapping, independent_parameters: List[str]
):
    if len(independent_parameters) > 0:
        trial_id = "_".join(
            "{}-{!s}".format(k, trial_parameters[k]) for k in independent_parameters
        )
        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return trial_id


def _get_object_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        pass

    raise ValueError(f"Unable to determine the name of {obj}")


class CallException(Exception):
    def __init__(self, func, args, kwargs, trial: "Trial"):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.trial = trial

    def __str__(self):
        return f"Error calling {self.func} (args={self.args}, kwargs={self.kwargs}) with {self.trial}"


Snapshot = namedtuple(
    "Snapshot",
    ["name", "resume_fn", "args", "kwargs"],
)


class Trial(collections.abc.MutableMapping):
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

    # Overwrite get to save supplied default value
    def get(self, key, default=None):
        return self.setdefault(key, default)

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
        trials_data = self._trial.store.match(
            experiment=name, resolved_parameters=dict(self)
        )

        if len(trials_data) == 1:
            trial_data = trials_data.pop()
            return Trial(trial_data)
        elif len(trials_data) > 1:
            msg = "Multiple matching parent experiments: " + ", ".join(
                trials_data.keys()
            )
            raise ValueError(msg)

        msg = "Trial has no attribute: {}".format(name)
        raise AttributeError(msg)

    def __repr__(self):
        return f"<Trial({dict(self)})>"

    def record_defaults(self, func: Callable, **defaults):
        """
        Record default parameters from a function and additional parameters.

        Args:
            func (callable): The keyword arguments of this function will be recorded if not already present.
            **kwargs: Additional arguments that will be recorded if not already present.

        Use :py:class:`functools.partial` to pass keyword parameters to `func` that should not be recorded.
        """

        __tracebackhide__ = True

        if not callable(func):
            raise ValueError("Only callables may be passed as first argument.")

        # Ignore partial keyword arguments
        try:
            partial_keywords = func.keywords  # type: ignore
        except AttributeError:
            partial_keywords = set()

        func_defaults = {
            param.name: param.default
            for param in inspect.signature(func).parameters.values()
            if param.name not in partial_keywords
        }

        # First set explicit defaults
        for name, value in defaults.items():
            if func_defaults is not None and name not in func_defaults:
                raise TypeError(f"{func} got an unexpected keyword argument '{name}'")

            self.setdefault(name, value)

        # Second, set remaining func defaults
        if func_defaults is not None:
            self.setdefaults(
                {
                    k: v
                    for k, v in func_defaults.items()
                    if v is not inspect.Parameter.empty
                }
            )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call the function applying the configured parameters.

        Args:
            func (callable): Function to be called.
            *args: Positional arguments to the function.
            **kwargs: Named defaults for the function.

        Returns:
            The return value of the function.

        The default values of the function are determined using :py:func:`inspect.signature`.
        Additional defaults can be given using ``**kwargs``.
        These defaults are recorded into the trial.

        As all default values are recorded, make sure that these have simple
        YAML-serializable types.

        Use :py:class:`functools.partial` to pass keyword parameters that should not be recorded.
        """

        # Record default parameters
        self.record_defaults(func, **kwargs)

        # Ignore partial keyword arguments
        try:
            partial_keywords = func.keywords  # type: ignore
        except AttributeError:
            partial_keywords = set()

        signature = inspect.signature(func)

        # Apply
        # Parameter names that can be given to the callable
        callable_names = set(
            param.name
            for param in signature.parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            and param.name not in partial_keywords
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )

        # Parameter names that have to be given to the callable
        required_names = set(
            param.name
            for param in signature.parameters.values()
            if param.default == param.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )

        # Does the callable accept kwargs?
        kwargs_present = any(
            param.kind == param.VAR_KEYWORD for param in signature.parameters.values()
        )

        if kwargs_present:
            parameters = dict(self)
        else:
            parameters = {k: v for k, v in self.items() if k in callable_names}

        # Bind known arguments and calculate missing
        bound_arguments = signature.bind_partial(*args, **parameters)
        missing_names = required_names - bound_arguments.arguments.keys()

        if missing_names:
            missing_names_str = ", ".join(f"'{n}'" for n in sorted(missing_names))
            missing_names_prefixed = ", ".join(
                f"'{self._prefix}{n}'" for n in sorted(missing_names)
            )

            raise TypeError(
                f"Missing required parameter(s) {missing_names_str} for {func}.\n"
                f"Supply {missing_names_prefixed} in your configuration."
            )

        try:
            return func(*args, **parameters)
        except Exception as exc:
            raise CallException(func, args, parameters, self) from exc

    def prefixed(self, prefix: str) -> "Trial":
        """
        Return new :py:class:`Trial` instance with prefix applied.

        Prefixes allow you to organize parameters and save keystrokes.
        """
        return Trial(self._trial, f"{self._prefix}{prefix}")

    def setdefaults(
        self, defaults: Union[Mapping, Iterable[Tuple[str, Any]], None] = None, **kwargs
    ):
        """
        Insert value in `defaults` into self it does not yet exist.

        Existing keys are not overwritten.
        If keyword arguments are given, the keyword arguments and their values are added.

        Parameters
        ----------
        defaults (Mapping or Iterable, optional): Default values.
        **kwargs: Additional default values.
        """

        itemiters = []

        if isinstance(defaults, Trial):
            itemiters.append(dict(defaults).items())
        elif isinstance(defaults, collections.abc.Mapping):
            itemiters.append(defaults.items())
        elif isinstance(defaults, collections.abc.Iterable):
            itemiters.append(defaults)  # type: ignore
        elif defaults is None:
            pass
        else:
            raise ValueError(f"Unexpected type for defaults: {type(defaults)}")

        itemiters.append(kwargs.items())

        for key, value in itertools.chain(*itemiters):
            self.setdefault(key, value)
        return self

    def choice(
        self,
        parameter_name: str,
        choices: Union[Mapping, Iterable],
        default=None,
    ):
        """
        Chose a value from an iterable whose name matches the value stored in parameter_name.

        If parameter_name is not configured, the first entry is returned and recorded as default.

        Args:
            parameter_name (str): Name of the parameter.
            choices (Mapping or Iterable): Mapping of names -> values or Iterable of values with a name (e.g. classes or functions).
            default: Default key in choices.

        Returns:
            The configured value from the iterable.

        """
        if isinstance(choices, collections.abc.Mapping):
            mapping = choices
        elif isinstance(choices, collections.abc.Iterable):
            names_values = [(_get_object_name(v), v) for v in choices]
            mapping = OrderedDict(names_values)
            if len(mapping) != len(names_values):
                raise ValueError("Duplicate names in {choices}")
        else:
            raise ValueError(f"Unexpected type of choices: {choices!r}")

        if default is not None:
            self.setdefault(parameter_name, default)

        entry_name = self[parameter_name]

        return mapping[entry_name]

    def flush(self):
        """Flush trial data to disk."""
        self._trial.save()

    def log(self, values, **kwargs):
        """
        Record metrics.

        Args:
            values (Mapping): Values to log.
        """
        values = {**values, **kwargs}
        self._trial.logger.log(values)

    def save_snapshot(
        self,
        name: str,
        resume_fn: Callable,
        *args,
        **kwargs,
    ):
        """
        Save a snapshot.

        All arguments must be picklable.

        Args:
            name (str, optional): Name of the snapshot.
            resume_fn (callable): Callable to resume from a saved state.
            args: Positional arguments provided to restore_fn.
            kwargs: Keyword arguments provided to restore_fn.

        Raises:
            TypeError: If the passed arguments do not match the signature of resume_fn.
        """

        # Check that supplied arguments are compatible with resume_fn to avoid unpleasent surprises
        signature = inspect.signature(resume_fn)
        # Raises a TypeError if the passed arguments do not match the signature of resume_fn
        signature.bind(*args, **kwargs)

        with atomicwrites.atomic_write(
            os.path.join(self.wdir, "snapshot"), mode="wb"
        ) as f:
            pickle.dump(Snapshot(name, resume_fn, args, kwargs), f)


def try_str(obj):
    try:
        return str(obj)
    except:  # pylint: disable=bare-except # noqa: E722
        return "<error>"


class TrialData:
    """
    Store data related to a trial.

    Arguments
        store: TrialStore
        data (optional): Trial data dictionary.
        func (optional): Experiment function.
    """

    def __init__(self, store: "TrialStore", data: Mapping, func=None):
        self.store = store
        self.data = data
        self.func = func

        self._validate_data()

        self.logger = YAMLLogger(self)

    def _validate_data(self):
        if "wdir" not in self.data:
            raise ValueError("data has to contain 'wdir'")
        if "id" not in self.data:
            raise ValueError("data has to contain 'id'")

    def run(self):
        """Run the current trial and save the results."""

        # Record intital state
        self.data["success"] = False
        self.data["time_start"] = datetime.datetime.now()
        self.data["result"] = None
        self.data["error"] = None

        try:
            result = self.func(Trial(self))
        except (Exception, KeyboardInterrupt) as exc:
            # Log complete exc to file
            error_fn = os.path.join(self.wdir, "error.txt")
            with open(error_fn, "w") as f:
                f.write(str(exc))
                f.write(traceback.format_exc())
                f.write("\n")
                for k, v in inspect.trace()[-1][0].f_locals.items():
                    f.write(f"{k}: {try_str(v)}\n")

            self.data["error"] = ": ".join(
                filter(None, (exc.__class__.__name__, str(exc)))
            )

            print("\n", flush=True)
            print(
                f"Error running {self.id}.\n"
                f"See {error_fn} for the complete traceback.",
                flush=True,
            )

            raise exc

        else:
            self.data["result"] = result
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
    def wdir(self):
        return self.data["wdir"]

    @property
    def is_failed(self):
        return self.data.get("error", None) is not None

    def remove(self):
        """Remove this trial from the store."""
        del self.store[self.id]


class TrialCollection(Collection):
    _missing = object()

    def __init__(self, trials: List[TrialData]):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __iter__(self):
        yield from self.trials

    def __contains__(self, trial: TrialData):
        return trial in self.trials

    def pop(self, index=-1):
        return self.trials.pop(index)

    @property
    def independent_parameters(self):
        independent_parameters = set()
        for t in self.trials:
            independent_parameters.update(
                t.data.get("experiment", {}).get("independent_parameters", [])
            )
        return independent_parameters

    @property
    def varying_parameters(self):
        """Independent parameters that vary in this trial collection."""
        independent_parameters = self.independent_parameters
        parameter_values = defaultdict(set)
        for t in self.trials:
            for p in independent_parameters:
                try:
                    v = t.data["parameters"][p]
                except KeyError:
                    parameter_values[p].add(self._missing)
                else:
                    parameter_values[p].add(v)

        return set(p for p in independent_parameters if len(parameter_values[p]) > 1)

    def to_pandas(self):
        import pandas as pd

        return pd.json_normalize([t.data for t in self.trials], max_level=1).set_index(
            "id"
        )

    def one(self):
        if len(self.trials) != 1:
            raise ValueError("No individual trial.")

        return self.trials[0]

    def filter(self, fn: Callable[[TrialData], bool]) -> "TrialCollection":
        """
        Return a filtered version of this trial collection.

        Args:
            fn (callable): A function that receives a TrialData instance and returns True if the trial should be kept.

        Returns:
            A new trial collection.
        """

        return TrialCollection(list(filter(fn, self.trials)))


class TrialStore(collections.abc.MutableMapping):
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> TrialCollection:
        func = _callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            experiment = experiment.name

        trials = []
        for trial in self.values():
            experiment_ = trial.data.get("experiment", {})
            if func is not None and _callable_to_name(experiment_.get("func")) != func:
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
                "func": _callable_to_name(experiment.func),
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
