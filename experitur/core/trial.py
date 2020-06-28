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
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import yaml

from experitur.core.logger import LoggerBase, YAMLLogger
from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict
from experitur.util import callable_to_name

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment

T = TypeVar("T")


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
        self, parameter_name: str, choices: Union[Mapping, Iterable], default=None,
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
