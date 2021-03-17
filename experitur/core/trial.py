import collections.abc
import inspect
import itertools
from collections import OrderedDict, defaultdict
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


from experitur.core.logger import YAMLLogger

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment
    from experitur.core.trial_store import TrialStore

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


class Trial(collections.abc.MutableMapping):
    """
    Data related to a trial.

    Args:
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

    def __init__(self, data: Mapping, store: "TrialStore", prefix: str = ""):
        self._store = store
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

    def get(self, key, default=None):
        """Get a parameter value.

        If key is not present, it is initialized with the provided default, just like :py:meth:`Trial.setdefault`.
        """
        return self.setdefault(key, default)

    def save(self):
        self._store[self.id] = self._data

    @property
    def is_failed(self):
        return self._data.get("error", None) is not None

    def remove(self):
        """Remove this trial from the store."""
        del self._store[self.id]

    def get_result(self, name):
        result = self._data["result"]
        if result is None:
            return None

        return result.get(name, None)

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
            raise type(exc)(
                f"Error calling {func} (args={args}, kwargs={kwargs}) with {self}"
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
        return Trial(self._data, self._store, f"{self._prefix}{prefix}")

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

    def log(self, values, **kwargs):
        """
        Record metrics.

        Args:
            values (Mapping): Values to log.
            **kwargs: Further values.
        """
        values = {**values, **kwargs}
        self._logger.log(values)


class TrialCollection(Collection):
    _missing = object()

    def __init__(self, trials: List[Trial]):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __iter__(self):
        yield from self.trials

    def __contains__(self, trial: Trial):
        return trial in self.trials

    def pop(self, index=-1):
        return self.trials.pop(index)

    @property
    def independent_parameters(self):
        independent_parameters = set()
        for t in self.trials:
            independent_parameters.update(
                getattr(t, "experiment", {}).get("independent_parameters", [])
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
                    v = t[p]
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

    def filter(self, fn: Callable[[Trial], bool]) -> "TrialCollection":
        """
        Return a filtered version of this trial collection.

        Args:
            fn (callable): A function that receives a Trial instance and returns True if the trial should be kept.

        Returns:
            A new trial collection.
        """

        return TrialCollection(list(filter(fn, self.trials)))
