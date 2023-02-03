import collections.abc
import copy
import datetime
import glob
import inspect
import itertools
import operator
import os
import os.path
import random
import socket
import warnings
from collections import OrderedDict, defaultdict
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import joblib
import numpy as np

from experitur.core.logger import YAMLLogger
from experitur.optimization import Objective, Optimization
from experitur.util import callable_to_name, freeze

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment
    from experitur.core.root_trial_collection import RootTrialCollection

T = TypeVar("T")


def _get_object_name(obj):
    if obj is None:
        return "None"

    try:
        return obj.__name__
    except AttributeError:
        pass

    try:
        return obj.__class__.__name__
    except AttributeError:
        pass

    raise ValueError(f"Unable to determine the name of {obj}")


def _to_str(obj):
    if isinstance(obj, str):
        return obj

    return repr(obj)


class Trial(collections.abc.MutableMapping):
    """
    Data related to a trial.

    This is automatically instanciated by experitur and provided to the experiment function:

    .. code-block:: python

        @Experiment(configurator={"a": [1,2,3], "prefix_a": [10]})
        def exp1(trial: Trial):
            # Access current value of parameter `a` (item access)
            trial["a"]

            # Access extra data (attribute access)
            trial.id # Trial ID
            trial.wdir # Trial working directory

            def func(a=1, b=2):
                ...

            # Record default trial of `func`
            trial.record_defaults(func)

            # Call `func` with current value of parameter `a` and `b`=5
            trial.call(func, b=5)

            # Access only trial starting with a certain prefix
            trial_prefix = trial.prefix("prefix_")

            # All the above works as expected:
            # trial_prefix.<attr>, trial_prefix[<key>], trial_prefix.record_defaults, trial_prefix.call, ...
            # In our case, trial_prefix["a"] == 10.
    """

    # Provided by _data:
    used_parameters: list
    id: str
    wdir: str
    resolved_parameters: Dict

    def __init__(
        self,
        data: MutableMapping,
        root: "RootTrialCollection",
        prefix: str = "",
        record_used_parameters=False,
    ):
        self._root = root
        self._data = data
        self._prefix = prefix
        self._record_used_parameters = record_used_parameters

        self._data.setdefault("used_parameters", [])

        self._valid = True

        if self._validate_data():
            self._logger = YAMLLogger(self)

    def _validate_data(self):
        for field in ("resolved_parameters", "wdir", "id"):
            if field not in self._data:
                print(f"ERROR: data has to contain '{field}', got {self._data!r}")
                self._valid = False

        return self._valid

    # MutableMapping provides concrete generic implementations of all
    # methods except for __getitem__, __setitem__, __delitem__,
    # __iter__, and __len__.

    def __getitem__(self, name):
        """Get the value of a parameter."""

        key = f"{self._prefix}{name}"
        if self._record_used_parameters:
            self.used_parameters.append(key)
        return self._data["resolved_parameters"][key]

    @property
    def unused_parameters(self):
        return sorted(set(self.resolved_parameters.keys()) - set(self.used_parameters))

    def __setitem__(self, name, value):
        """Set the value of a parameter."""
        self._data["resolved_parameters"][f"{self._prefix}{name}"] = value

    def __delitem__(self, name):
        """Delete a parameter."""
        del self._data["resolved_parameters"][f"{self._prefix}{name}"]

    def __iter__(self):
        start = len(self._prefix)
        return (
            k[start:]
            for k in self._data["resolved_parameters"]
            if k.startswith(self._prefix)
        )

    def todict(self, with_prefix=False):
        """
        Convert trial data to dictionary.

        Args:
            with_prefix (bool): Use full key, even when using under `Trial.prefixed`.
        """

        data = dict(self)
        if with_prefix:
            data = {self._prefix + k: v for k, v in data.items()}
        return data

    def __len__(self):
        return sum(1 for k in self)

    def __repr__(self):
        return f"<Trial({dict(self)})>"

    def descr(
        self,
        trial_collection: Optional["TrialCollection"] = None,
        ignore=None,
        only=None,
        drop_prefixes=None,
        status=False,
        replace=None,
        include=None,
    ):
        parameters = dict(self)

        if ignore is None:
            ignore = set()

        if include is None:
            include = set()

        if drop_prefixes is None:
            drop_prefixes = []

        if replace is None:
            replace = {}

        def _drop_prefix(k: str):
            for prefix in drop_prefixes:
                if k.startswith(prefix):
                    return k[len(prefix) :]
            return k

        if trial_collection is not None:
            varying_parameters = trial_collection.varying_parameters
            parameters = {
                _drop_prefix(replace.get(k, k)): v
                for k, v in parameters.items()
                if (only is not None and k in only)
                or (only is None and k in varying_parameters and k not in ignore)
                or (k in include)
            }

        descr = ", ".join(f"{k}={_to_str(v)}" for k, v in sorted(parameters.items()))

        if not descr:
            descr = "[default]"

        # Status indicator
        status_chr = _trial_status_chr(self, success="")
        if status and status_chr:
            descr = f"{status_chr} {descr}"

        return descr

    def get(self, key, default=None, setdefault=True):
        """Get a parameter value.

        If key is not present, it is initialized with the provided default, just like :py:meth:`Trial.setdefault`.
        """
        if setdefault:
            return self.setdefault(key, default)

        return super().get(key, default)

    def save(self):
        # Compact used parameters
        self.used_parameters = sorted(set(self.used_parameters))
        # Save unused parameters
        self._data["unused_parameters"] = self.unused_parameters

        # Write to the store
        self._root.update(self)

    def save_checkpoint(self, *args, **kwargs):
        """
        Save a checkpoint that allows the experiment to be resumed.

        Args:
            *args, **kwargs: Arguments supplied to the experiment function upon resumption.
        """

        # XXX: Interaction with log: Commit log entries if checkpoint is saved. Remove uncommitted log entries upon restorage.

        checkpoint = dict(args=args, kwargs=kwargs)

        checkpoint_fn = os.path.join(
            self.wdir, f"checkpoint_{datetime.datetime.now().isoformat()}.chk"
        )

        joblib.dump(checkpoint, checkpoint_fn, compress=True)

        try:
            old_checkpoint_fn = self.checkpoint_fn
        except AttributeError:
            old_checkpoint_fn = None

        self.checkpoint_fn = checkpoint_fn
        self.save()

        if old_checkpoint_fn:
            # TODO: Catch sensible exceptions
            os.remove(old_checkpoint_fn)

        self.log(save_checkpoint=checkpoint_fn)
        self._logger.commit()

        print(f"Saved checkpoint: {checkpoint_fn}")

    def load_checkpoint(self):
        try:
            checkpoint_fn = self.checkpoint_fn
        except AttributeError:
            checkpoint_fn = None

        if checkpoint_fn is None:
            return None

        self._logger.rollback()
        self.log(load_checkpoint=checkpoint_fn)
        return joblib.load(checkpoint_fn)

    @property
    def runtime(self) -> datetime.timedelta:
        time_start = self._data.get("time_start")
        if time_start is None:
            return datetime.timedelta()
        time_end = self._data.get("time_end") or datetime.datetime.now()
        return time_end - time_start

    @property
    def is_failed(self):
        return not self._valid or self._data.get("error", None) is not None

    @property
    def is_successful(self):
        return self._data.get("success", False)

    @property
    def is_zombie(self):
        if self.is_successful or self.is_failed:
            return False

        meta = self.experiment.get("meta", {})
        if meta.get("hostname") != socket.gethostname():
            return False

        pid = meta.get("pid")
        if pid is None:
            return False

        try:
            import psutil
        except ImportError:
            warnings.warn("psutil is not available, zombie trials can not be detected")
        else:
            if not psutil.pid_exists(pid):
                return True

        return False

    @property
    def is_resumable(self) -> bool:
        return hasattr(self, "checkpoint_fn") and bool(self.checkpoint_fn)

    @property
    def revision(self) -> Optional[str]:
        try:
            return self._data["revision"]
        except KeyError:
            return None

    @property
    def status(self):
        try:
            return self._data["status"]
        except KeyError:
            return None

    @status.setter
    def status(self, value):
        self._data["status"] = value

    def remove(self):
        """Remove this trial from the store."""
        self._root.remove(self)

    def get_result(self, name):
        result = self._data["result"]
        if result is None:
            return None

        return result.get(name, None)

    def update_result(self, values: Optional[Mapping] = None, **kwargs):
        if self._data["result"] is None:
            self._data["result"] = {}

        if values is None:
            values = kwargs
        else:
            values = {**values, **kwargs}

        self._data["result"].update(
            {f"{self._prefix}{k}": v for k, v in values.items()}
        )

    def update_used_parameters(self, parameters: Iterable):
        for p in parameters:
            self[p]

    def __getattr__(self, name: str):
        """Access extra attributes."""

        __tracebackhide__ = True  # pylint: disable=unused-variable

        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def record_defaults(self, func: Callable, **defaults):
        """
        Record default parameters from a function and additional parameters.

        Args:
            func (callable): The keyword arguments of this function will be recorded if not already present.
            **kwargs: Additional arguments that will be recorded if not already present.

        Use :py:class:`functools.partial` to pass keyword parameters to `func` that should not be recorded.
        """

        __tracebackhide__ = True  # pylint: disable=unused-variable

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

        # Overwrite explicit defaults
        for name, value in defaults.items():
            if name not in func_defaults:
                raise TypeError(f"{func} got an unexpected keyword argument '{name}'")

            func_defaults[name] = value

        # Filter parameters that are still empty
        func_defaults = {
            k: v for k, v in func_defaults.items() if v is not inspect.Parameter.empty
        }

        # Store recorded defaults
        self.setdefaults(func_defaults)

        return func_defaults

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

        If the called function throws an exception, an exception of the same type
        is thrown with additional information about the parameters.

        Use :py:class:`functools.partial` to pass hidden keyword parameters that should not be recorded.
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
            if param.default is param.empty
            and param.name not in partial_keywords
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
            print("signature", signature)
            print("callable_names", callable_names)
            print("parameters", parameters)
            print("bound_arguments", bound_arguments)
            raise type(exc)(
                f"Error calling {func} {signature} with args={args}, kwargs={parameters}, trial={self}"
            ) from exc

    def prefixed(self, prefix: str) -> "Trial":
        """
        Return new :py:class:`Trial` instance with prefix applied.

        Prefixes allow you to organize parameters and save keystrokes.

        Example:

            .. code-block:: python

                trial_prefix = trial.prefix("prefix_")
                trial_prefix["a"] == trial["prefix_a"] # True
        """
        return Trial(
            self._data,
            self._root,
            f"{self._prefix}{prefix}",
            self._record_used_parameters,
        )

    def setdefaults(
        self,
        defaults: Union["Trial", Mapping, Iterable[Tuple[str, Any]], None] = None,
        **kwargs,
    ):
        """
        Set multiple default values for parameters that do not yet exist.

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
        choices: Union[Mapping[Any, T], Iterable[T]],
        default=None,
    ) -> T:
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

        if default is None:
            # Select first item as default
            for default in mapping:
                break

        self.setdefault(parameter_name, default)

        entry_name = self[parameter_name]

        try:
            return mapping[entry_name]
        except KeyError:
            raise ValueError(
                f"Unknown option {entry_name!r}. Options are: {sorted(mapping.keys())!r}"
            )

    # TODO: Rework logging: channels, log to storage, ...
    def log(self, values=None, **kwargs):
        """
        Record metrics.

        Args:
            values (Mapping, optional): Values to log.
            **kwargs: Further values.
        """
        if values is not None:
            values = {**values, **kwargs}
        else:
            values = kwargs
        self._logger.log(values)

    def log_msg(self, msg: str):
        """
        Log a message.
        """

        raise NotImplementedError()

    def get_log(self, aggregate=True):
        if not aggregate:
            yield from self._logger.read()
            return

        acc = {}
        for entry in self._logger.read():
            if any(k in acc for k in entry.keys()):
                yield acc
                acc = entry.copy()
            else:
                acc.update(entry)
        # yield final entry
        yield acc

    def aggregate_log(self, include):
        return self._logger.aggregate(include)

    def find_files(self, pattern, recursive=False) -> List:
        """
        Find files of the trial.

        Args:
            pattern (str): Filename pattern.
            recursive (boolean, optional): Search recursively.
                If True, the pattern '**' will match any files and
                zero or more directories and subdirectories.

        Returns:
            List of filenames.
        """

        pattern = os.path.join(glob.escape(self.wdir), pattern)
        return glob.glob(pattern, recursive=recursive)

    def file(self, filename, make_parents=False):
        """
        Return filename relative to the trial's working directory.

        Args:
            make_parents (bool, optional): Make intermediary directories.
        """
        filename = os.path.join(self.wdir, filename)
        if make_parents:
            dirname = os.path.dirname(filename)
            os.makedirs(dirname, exist_ok=True)
        return filename

    def has_file(self, pattern, recursive=False):
        return len(self.find_files(pattern, recursive)) > 0

    def find_file(self, pattern: str, recursive: bool = False):
        """
        Find a file of the trial.

        Args:
            pattern (str): Filename pattern.
            recursive (boolean, optional): Search recursively.
                If True, the pattern '**' will match any files and
                zero or more directories and subdirectories.

        Returns:
            filename

        Raises:
            ValueError: If not exactly one file was found.
        """
        matches = self.find_files(pattern, recursive)
        if not matches:
            raise ValueError(f"No matches for {pattern}")
        if len(matches) > 1:
            raise ValueError(f"Too many matches for {pattern}: {matches}")

        return matches[0]

    def should_prune(self, default_values=None) -> bool:
        pruning_config = self._data.get("pruning_config", None)

        # If pruning is not configured, do not prune.
        if pruning_config is None:
            return False

        parameters = pruning_config["parameters"]
        step_name = pruning_config["step_name"]
        minimize = pruning_config["minimize"]
        invert_signs = pruning_config["invert_signs"]
        min_steps = pruning_config["min_steps"]
        min_count = pruning_config["min_count"]
        quantile = pruning_config["quantile"]

        def prep_entry(entry):
            if step_name not in entry or minimize not in entry:
                return None
            return entry[step_name], ((-1) ** invert_signs) * entry[minimize]

        orig_last_entry = last_entry = (
            {**default_values, **self._logger.last_entry}
            if default_values is not None
            else self._logger.last_entry
        )

        if not last_entry:
            raise RuntimeError("No log available for current trial.")

        last_entry = prep_entry(last_entry)

        if last_entry is None:
            available_fields = ", ".join(f"'{k}'" for k in orig_last_entry.keys())
            raise RuntimeError(
                f"Log of the current trial does not contain '{step_name}' and/or '{minimize}'. Available fields: {available_fields}"
            )

        own_max_step, own_last_metric = last_entry

        # If this trial ran less than min_steps, do not prune.
        if own_max_step < min_steps:
            return False

        comparison_trials = self._root.match(resolved_parameters=parameters).filter(
            lambda trial: trial.id != self.id
        )

        surviving_trials = 0
        best_metrics_sofar = []
        for trial in comparison_trials:
            log = list(
                filter(None, (prep_entry(e) for e in trial.get_log(aggregate=True)))
            )

            if not log:
                continue

            best_metrics_sofar.append(min(e[1] for e in log if e[0] <= own_max_step))
            trial_max_step = max(e[0] for e in log)

            if trial_max_step >= own_max_step:
                surviving_trials += 1

        # If at this point less than min_count are surviving, do not prune.
        if surviving_trials < min_count:
            return False

        # If this trial is currently better than the specified quantile, do not prune.
        if own_last_metric <= np.quantile(best_metrics_sofar, quantile):
            return False

        # TODO: patience

        return True

    def copy(self):
        """
        Create a copy of the Trial instance.

        While the data is deep-copied, the ID is the same, so saving overwrites the original data.
        """

        return Trial(
            copy.deepcopy(self._data),
            self._root,
            self._prefix,
            self._record_used_parameters,
        )


def _normalize_runtime(runtime: Optional[datetime.timedelta]):
    if runtime is None:
        runtime = datetime.timedelta()

    # 1s resolution
    return datetime.timedelta(seconds=runtime // datetime.timedelta(seconds=1))


Trial.get


class BaseTrialCollection(Collection[Trial]):
    class _Missing:
        def __repr__(self):
            return "<missing>"

    _missing = _Missing()

    def __init__(
        self, independent_parameters_include=None, independent_parameters_exclude=None
    ):
        self._independent_parameters_include = independent_parameters_include or set()
        self._independent_parameters_exclude = independent_parameters_exclude or set()

    def update_independent_parameters(self, include=None, exclude=None):
        if include is not None:
            self._independent_parameters_include.update(include)
        if exclude is not None:
            self._independent_parameters_exclude.update(exclude)

    @property
    def independent_parameters(self) -> Set[str]:
        independent_parameters = set()
        for t in self:
            independent_parameters.update(
                getattr(t, "experiment", {}).get("independent_parameters", [])
            )
        return independent_parameters.union(
            self._independent_parameters_include
        ).difference(self._independent_parameters_exclude)

    @property
    def varying_parameters(self):
        """Parameters that vary in this trial collection."""
        return {k: v for k, v in self.parameters.items() if len(v) > 1}

    @property
    def parameters(self):
        parameter_values = defaultdict(set)
        all_parameters = set()
        for t in self:
            all_parameters.update(t.keys())

        for t in self:
            for p in all_parameters:
                try:
                    v = t[p]
                except KeyError:
                    parameter_values[p].add(self._missing)
                else:
                    # FIXME: Instead of freeze, use a list
                    parameter_values[p].add(freeze(v))

        return parameter_values

    @property
    def results(self):
        return {t.id: t.result for t in self}

    @property
    def invariant_parameters(self):
        """Independent parameters that do not vary in this trial collection."""
        independent_parameters = self.independent_parameters
        parameter_values = defaultdict(set)
        for t in self:
            for p in independent_parameters:
                try:
                    v = t[p]
                except KeyError:
                    parameter_values[p].add(self._missing)
                else:
                    # FIXME: Instead of freeze, use a list
                    parameter_values[p].add(freeze(v))

        return {
            p: parameter_values[p].pop()
            for p in independent_parameters
            if len(parameter_values[p]) == 1
        }

    def to_pandas(self, full=True):
        if not self:
            raise ValueError("Empty trial collection.")

        import pandas as pd

        if full:
            return pd.json_normalize([t._data for t in self], max_level=1).set_index(
                "id"
            )

        return pd.DataFrame(
            [{"id": t.id, **dict(t), **t.result} for t in self]
        ).set_index("id")

    def one(self) -> Trial:
        if len(self) != 1:
            raise ValueError(f"No individual trial (Found {len(self)})")

        return next(iter(self))

    def filter(self, fn: Callable[[Trial], bool]) -> "TrialCollection":
        """
        Return a filtered version of this trial collection.

        Args:
            fn (callable): A function that receives a Trial instance and returns True if the trial should be kept.

        Returns:
            A new trial collection.
        """

        result = TrialCollection(
            [t for t in self if fn(t)],
            independent_parameters_include=self._independent_parameters_include,
            independent_parameters_exclude=self._independent_parameters_exclude,
        )

        return result

    def shuffle(self) -> "TrialCollection":
        """
        Return a shuffled version of this trial collection.

        Returns:
            A new trial collection.
        """

        trials = [t for t in self]
        random.shuffle(trials)

        return TrialCollection(
            trials,
            independent_parameters_include=self._independent_parameters_include,
            independent_parameters_exclude=self._independent_parameters_exclude,
        )

    def to_minimization(
        self,
        minimize: Objective = None,
        maximize: Objective = None,
        include_na=True,
        quantile=1.0,
    ) -> List[Tuple[Trial, Real]]:
        """
        Turn trial collection into a minimization problem.

        This can be used to find the best trial or inside optimization routines.

        Args:
            minimize: Name or list of names of trial results to minimize.
            maximize: Name or list of names of trial results to minimize.
        """

        optimization = Optimization(minimize=minimize, maximize=maximize)

        trials = [t for t in self]
        results = [trial.result if trial.result else None for trial in trials]
        results = optimization.to_minimization(results, quantile=quantile)

        if include_na:
            return list(zip(trials, results))

        return [(t, r) for t, r in zip(trials, results) if r is not None]

    def pareto_optimal(
        self, minimize: Objective = None, maximize: Objective = None
    ) -> "TrialCollection":
        """Filter trials that are pareto-optimal."""

        optimization = Optimization(minimize=minimize, maximize=maximize)

        trials = [t for t in self]
        n_dominated = optimization.n_dominated(trials)

        return TrialCollection(
            [t for t, d in zip(trials, n_dominated) if d == 0],
            independent_parameters_include=self._independent_parameters_include,
            independent_parameters_exclude=self._independent_parameters_exclude,
        )

    def groupby(
        self,
        parameters: Union[List[str], str, None] = None,
        experiment=False,
    ) -> "TrialCollectionGroupby":
        if isinstance(parameters, str):
            parameters = [parameters]

        if not experiment and parameters is None:
            return TrialCollectionGroupby({})

        def make_key(trial: Trial):
            key = {}
            if parameters is not None:
                key.update({p: freeze(trial.get(p)) for p in parameters})
            if experiment:
                key["__experiment"] = trial.experiment["name"]

            return frozenset(key.items())

        groups = defaultdict(TrialCollection)
        for trial in self:
            groups[make_key(trial)].append(trial)

        return TrialCollectionGroupby(groups)

    def __str__(self):
        return self.format()

    def format(self, process_info=True, status=True, time=True, error=True, descr=True):
        """
        Format TrialCollection.

        Args:
            process_info: Show process information like hostname and PID.
        """

        result = []

        if len(self):
            result.append(f"{len(self)} trials:")
        else:
            result.append("0 trials")

        for i, trial in enumerate(self):
            result.append(f"{i:2d}: {trial.id}")

            if descr:
                trial_descr = trial.descr(self)
                if trial_descr:
                    result.append(f"    {trial_descr}")

            if process_info:
                hostname = trial.experiment["meta"].get("hostname", "<no host>")
                pid = str(trial.experiment["meta"].get("pid", "<no pid>"))
                result.append(f"    {hostname}:{pid}")

            if time:
                runtime = trial.runtime
                if runtime is not None:
                    result.append(f"    {_normalize_runtime(runtime)!s}")

            if status:
                status_chr = _trial_status_chr(trial)
                status_long = {
                    "!": "Failed",
                    "+": "Successful",
                    ">": "Running",
                    "Z": "Zombie",
                }[status_chr]

                if error and trial.is_failed:
                    status_long = status_long + ": " + str(trial.error)
                elif trial.status:
                    status_long = status_long + ": " + str(trial.status)

                result.append(f"    {status_chr} {status_long}")

        return "\n".join(result)

    def print(self, process_info=True, status=True, time=True, error=True, descr=True):
        print(
            self.format(
                process_info=process_info,
                status=status,
                time=time,
                error=error,
                descr=descr,
            )
        )

    def _repr_html_(self):
        varying_parameters = sorted(self.varying_parameters.keys())

        output = [
            "<table>",
            "<thead>",
            "<tr>",
            "<th></th>",
            "<th></th>",
            "<th>ID</th>",
            "<th>Date</th>",
            "<th>Runtime</th>",
            "<th>Process</th>",
            "<th>Status</th>",
        ]

        output.extend(f"<th>{p}</th>" for p in varying_parameters)

        output.extend(
            [
                "</tr>",
                "</thead>",
                "<tbody>",
            ]
        )

        for i, t in enumerate(self):
            status = _trial_status_chr(t)
            output.append("<tr>")
            output.append(f"<th>{i}</th>")
            output.append(f"<th>{status}</th>")
            output.append(f"<th>{t.id}</th>")
            time_start = t._data.get("time_start")
            time_start = "" if time_start is None else "{:%Y-%m-%d}".format(time_start)
            output.append(f"<th>{time_start}</th>")
            output.append(f"<td>{_normalize_runtime(t.runtime)!s}</td>")
            hostname = t.experiment["meta"].get("hostname", "")
            pid = str(t.experiment["meta"].get("pid", ""))
            output.append(f"<td>{hostname}:{pid}</td>")
            output.append(f"<td>{t.status if t.status else ''}</td>")
            output.extend(
                f"<td>{t.get(p, '', setdefault=False)}</td>" for p in varying_parameters
            )
            output.append("</tr>")

        output.extend(
            [
                "</tbody>",
                "</table>",
            ]
        )

        return "\n".join(output)

    def best_n(
        self, n, minimize: Objective = None, maximize: Objective = None
    ) -> "TrialCollection":
        trial_results = self.to_minimization(
            minimize=minimize, maximize=maximize, include_na=False
        )
        trial_results.sort(key=operator.itemgetter(1))

        return TrialCollection([tr[0] for tr in trial_results[:n]])

    def sorted(self, key: Optional[Callable] = None, reverse=False):
        if key is None:
            key = operator.attrgetter("id")
        return TrialCollection(sorted(self, key=key, reverse=reverse))

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> "TrialCollection":
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            if experiment.name is None:
                raise ValueError(f"Experiment {experiment!r} has no name set")
            experiment = experiment.name

        from experitur.core.trial_store import _match_parameters

        trial_list: List[Trial] = []
        for trial in self:
            if (
                func is not None
                and callable_to_name(trial.experiment.get("func")) != func
            ):
                continue

            if parameters is not None and not _match_parameters(
                parameters, trial.parameters
            ):
                continue

            if resolved_parameters is not None and not _match_parameters(
                resolved_parameters, trial.resolved_parameters
            ):
                continue

            if experiment is not None and trial.experiment.get("name") != experiment:
                continue

            trial_list.append(trial)

        return TrialCollection(trial_list)

    def failed(self):
        return self.filter(lambda trial: trial.is_failed)

    def successful(self):
        return self.filter(lambda trial: trial.is_successful)


class TrialCollection(MutableSequence[Trial], BaseTrialCollection):
    def __init__(
        self,
        trials: Optional[Iterable[Trial]] = None,
        independent_parameters_include=None,
        independent_parameters_exclude=None,
    ):
        BaseTrialCollection.__init__(
            self,
            independent_parameters_include=independent_parameters_include,
            independent_parameters_exclude=independent_parameters_exclude,
        )

        if trials is None:
            trials = []
        else:
            trials = list(trials)
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    @overload
    def __getitem__(self, index: slice) -> "TrialCollection":
        ...

    @overload
    def __getitem__(self, index: int) -> Trial:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TrialCollection(
                self.trials[index],
                independent_parameters_include=self._independent_parameters_include,
                independent_parameters_exclude=self._independent_parameters_exclude,
            )

        if isinstance(index, str):
            try:
                return self.filter(lambda trial: trial.id == index).one()
            except ValueError:
                raise KeyError(index) from None

        return self.trials[index]

    def __iter__(self):
        yield from self.trials

    def __contains__(self, trial: Trial):
        return trial in self.trials

    def __add__(self, other):
        return TrialCollection(self.trials + other.trials)

    def __delitem__(self, index):
        del self.trials[index]

    def __setitem__(self, index, o):
        self.trials[index] = o

    def pop(self, index=-1):
        return self.trials.pop(index)

    def insert(self, index, trial: Trial):
        self.trials.insert(index, trial)

    def setdefaults(self, values=None, **kwargs):
        """Set multiple default values for parameters of contained trials."""
        values = kwargs if values is None else {**values, **kwargs}
        for t in self:
            t.setdefaults(values)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.trials!r})"


class TrialCollectionGroupby(collections.abc.Sized, collections.abc.Iterable):
    def __init__(self, groups: Mapping[Any, TrialCollection]):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        for key, group in self.groups.items():
            yield dict(key), group

    def __repr__(self):
        return f"<TrialCollectionGroupby {list(self.groups.items())}>"

    def filter(self, fn: Callable[[Trial], bool]) -> "TrialCollectionGroupby":
        """
        Apply a filter to each of the groups.

        Args:
            fn (callable): A function that receives a Trial instance and returns True if the trial should be kept.

        Returns:
            A new TrialCollectionGroupby.
        """

        return TrialCollectionGroupby({k: v.filter(fn) for k, v in self.groups.items()})

    def best_n(self, n, minimize: Objective = None, maximize: Objective = None):
        return TrialCollectionGroupby(
            {k: v.best_n(n, minimize, maximize) for k, v in self.groups.items()}
        )

    def pareto_optimal(self, minimize: Objective = None, maximize: Objective = None):
        return TrialCollectionGroupby(
            {
                k: v.pareto_optimal(minimize=minimize, maximize=maximize)
                for k, v in self.groups.items()
            }
        )

    def coalesce(self):
        """
        Coalesce the individual groups into one TrialCollection.
        """

        trials = []
        for group in self.groups.values():
            trials.extend(group)

        return TrialCollection(trials)

    def keys(self):
        for key in self.groups.keys():
            yield dict(key)

    def apply(self, func: Callable[[Dict, TrialCollection], TrialCollection]):
        """Apply function func group-wise and combine the results together."""
        trials = []
        for key, group in self.groups.items():
            trials.extend(func(dict(key), group))

        return TrialCollection(trials)


def _trial_status_chr(trial: Trial, success="+"):
    if trial.is_failed:
        return "!"  # Failed

    if trial.is_zombie:
        return "Z"

    if getattr(trial, "success", False):
        return success  # Successful

    return ">"  # Running
