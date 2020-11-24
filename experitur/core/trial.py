import collections.abc
import glob
import inspect
import itertools
import os.path
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from experitur.core.logger import YAMLLogger
from experitur.util import callable_to_name, freeze

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment

T = TypeVar("T")


def _get_object_name(obj):
    try:
        return obj.__name__
    except AttributeError:
        pass

    try:
        return obj.__class__.__name__
    except AttributeError:
        pass

    raise ValueError(f"Unable to determine the name of {obj}")


class CallException(Exception):
    def __init__(self, func, args, kwargs, trial: "Trial"):
        super().__init__(
            f"Error calling {func} (args={args}, kwargs={kwargs}) with {trial}"
        )
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.trial = trial


class Trial(collections.abc.MutableMapping):
    """
    Data related to a trial.

    Arguments
        store: TrialStore
        data (optional): Trial data dictionary.
        func (optional): Experiment function.
    
    This is automatically instanciated by experitur and provided to the experiment function:

    .. code-block:: python

        @Experiment(parameters={"a": [1,2,3], "prefix_a": [10]})
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

    def __init__(
        self, data: MutableMapping, root: "RootTrialCollection", prefix: str = "",
    ):
        self._root = root
        self._data = data
        self._prefix = prefix

        self._validate_data()

        self._logger = YAMLLogger(self)

    def _validate_data(self):
        if "wdir" not in self._data:
            raise ValueError("data has to contain 'wdir'")
        if "id" not in self._data:
            raise ValueError("data has to contain 'id'")

    # MutableMapping provides concrete generic implementations of all
    # methods except for __getitem__, __setitem__, __delitem__,
    # __iter__, and __len__.

    def __getitem__(self, name):
        """Get the value of a parameter."""
        return self._data["resolved_parameters"][f"{self._prefix}{name}"]

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

    def __len__(self):
        return sum(1 for k in self)

    def __repr__(self):
        return f"<Trial({dict(self)})>"

    # Overwrite get to save supplied default value
    def get(self, key, default=None):
        return self.setdefault(key, default)

    def save(self):

        # Write to the store
        self._root.update(self)

    @property
    def is_failed(self):
        return self._data.get("error", None) is not None

    @property
    def is_successful(self):
        return self._data.get("success", False)

    def remove(self):
        """Remove this trial from the store."""
        self._root.remove(self)

    def get_result(self, name):
        result = self._data["result"]
        if result is None:
            return None

        return result.get(name, None)

    def __getattr__(self, name: str):
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
        return Trial(self._data, self._root, f"{self._prefix}{prefix}")

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

    def log(self, values, **kwargs):
        """
        Record metrics.

        Args:
            values (Mapping): Values to log.
        """
        values = {**values, **kwargs}
        self._logger.log(values)


class BaseTrialCollection(collections.abc.Collection):
    class _Missing:
        def __repr__(self):
            return "<missing>"

    _missing = _Missing()

    @property
    def independent_parameters(self):
        independent_parameters = set()
        for t in self:
            independent_parameters.update(
                getattr(t, "experiment", {}).get("independent_parameters", [])
            )
        return independent_parameters

    @property
    def varying_parameters(self):
        """Independent parameters that vary in this trial collection."""
        independent_parameters = self.independent_parameters
        parameter_values = defaultdict(set)
        for t in self:
            for p in independent_parameters:
                try:
                    v = t[p]
                except KeyError:
                    parameter_values[p].add(self._missing)
                else:
                    parameter_values[p].add(freeze(v))

        return {
            p: parameter_values[p]
            for p in independent_parameters
            if len(parameter_values[p]) > 1
        }

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
                    parameter_values[p].add(freeze(v))

        return {
            p: parameter_values[p].pop()
            for p in independent_parameters
            if len(parameter_values[p]) == 1
        }

    def to_pandas(self):
        import pandas as pd

        tdata = [t._data for t in self]

        if not tdata:
            raise ValueError("Empty trial collection.")

        try:
            return pd.json_normalize(tdata, max_level=1).set_index("id")
        except:
            print("Can't convert to pandas:", tdata)
            raise

    def one(self):
        if len(self) != 1:
            raise ValueError("No individual trial.")

        return next(iter(self))

    def filter(self, fn: Callable[[Trial], bool]) -> "TrialCollection":
        """
        Return a filtered version of this trial collection.

        Args:
            fn (callable): A function that receives a Trial instance and returns True if the trial should be kept.

        Returns:
            A new trial collection.
        """

        result = TrialCollection([t for t in self if fn(t)])

        return result

    def groupby(
        self, parameters=None, experiment=False,
    ) -> Generator[Tuple[dict, "TrialCollection"], None, None]:
        if isinstance(parameters, str):
            parameters = [parameters]

        if not experiment and parameters is None:
            return TrialCollectionGroupby({})

        def make_key(trial: Trial):
            key = {}
            if parameters is not None:
                key.update({p: trial.get(p) for p in parameters})
            if experiment:
                key["__experiment"] = trial.experiment["name"]

            return frozenset(key.items())

        groups = defaultdict(TrialCollection)
        for trial in self:
            groups[make_key(trial)].append(trial)

        return TrialCollectionGroupby(groups)

    def __str__(self):
        return self.format()

    def format(self, process_info=True, status=True, time=True):
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
            descr = trial.descr(self)
            if descr:
                result.append(f"    {trial.descr(self)}")

            if process_info:
                hostname = trial.experiment["meta"].get("hostname", "<no host>")
                pid = str(trial.experiment["meta"].get("pid", "<no pid>"))
                result.append(f"    {hostname}:{pid}")

            if time:
                runtime = trial.runtime
                if runtime is not None:
                    result.append(f"    {runtime!s}")

            if status:
                status_chr = _trial_status_chr(trial)
                status_long = {
                    "!": "Failed",
                    "+": "Successful",
                    ">": "Running",
                    "Z": "Zombie",
                }[status_chr]
                result.append(f"    {status_chr} {status_long}")

        return "\n".join(result)

    def print(self):
        print(self.format())

    def sorted(self, key=None, reverse=False):
        if key is None:
            key = operator.attrgetter("id")
        return TrialCollection(sorted(self, key=key, reverse=reverse))

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> List[Dict]:
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            if experiment.name is None:
                raise ValueError(f"Experiment {experiment!r} has no name set")
            experiment = experiment.name

        from experitur.core.trial_store import _match_parameters

        trial_data_list = []
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

            trial_data_list.append(trial)

        return TrialCollection(trial_data_list)


class TrialCollection(collections.abc.MutableSequence, BaseTrialCollection):
    def __init__(self, trials: Optional[Iterable[Trial]] = None):
        if trials is None:
            trials = []
        else:
            trials = list(trials)
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TrialCollection(self.trials[index])

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

    def coalesce(self):
        """
        Coalesce the individual groups into one TrialCollection.
        """

        trials = []
        for group in self.groups.values():
            trials.extend(group)

        return TrialCollection(trials)

