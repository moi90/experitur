import copy
import datetime
import functools
import itertools
import os
import pdb
import socket
import traceback
import warnings
from collections import defaultdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import click

from experitur.core.configurators import (
    BaseConfigurator,
    Configurable,
    MultiplicativeConfiguratorChain,
    is_invariant,
    validate_configurators,
)
from experitur.core.context import ExperimentStoppedError, get_current_context
from experitur.core.trial import Trial
from experitur.core.trial_store import ModifiedError
from experitur.errors import ExperiturError
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict
from experitur.util import (
    callable_to_name,
    clean_unset,
    cprint,
    ensure_dict,
    ensure_list,
)

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.context import Context
    from experitur.core.trial import TrialCollection


def try_str(obj):
    try:
        return str(obj)
    except:  # pylint: disable=bare-except # noqa: E722
        return "<error>"


class ExperimentError(ExperiturError):
    pass


class StopExecution(ExperimentError):
    pass


class CommandNotFoundError(ExperimentError):
    pass


class TrialNotFoundError(ExperimentError):
    pass


class SkipTrial(Exception):
    pass


def format_trial_parameters(func=None, parameters=None, experiment=None):
    if func is not None:
        try:
            func = func.__name__
        except AttributeError:
            func = str(func)
    else:
        func = "_"

    if parameters is not None:
        parameters = (
            "("
            + (", ".join("{}={}".format(k, repr(v)) for k, v in parameters.items()))
            + ")"
        )
    else:
        parameters = "()"

    if experiment is not None:
        func = "{}:{}".format(str(experiment), func)

    return func + parameters


def _detect_keyboard_interrupt(exception: BaseException):
    queue = [exception]
    while queue:
        exception = queue.pop()

        if isinstance(exception, KeyboardInterrupt):
            return True

        if exception.__cause__ is not None:
            queue.append(exception.__cause__)
        if exception.__context__ is not None:
            queue.append(exception.__context__)
    return False


class _SkipCache:
    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment

        self._trial_collection: Optional["TrialCollection"] = None

    def update(self):
        self._trial_collection = self.experiment.ctx.trials.match(
            func=self.experiment.func
        )

    def get_trials(self, parameters):
        if self._trial_collection is not None:
            existing = self._trial_collection.match(resolved_parameters=parameters)
            if existing:
                return existing

        # If configuration was not yet found, update cache...
        self.update()

        # ... and retry
        return self._trial_collection.match(resolved_parameters=parameters)  # type: ignore

    def __contains__(self, parameters):
        existing_trials = self.get_trials(parameters)

        if existing_trials:
            return True

        return False


class Experiment(Configurable):
    """
    Define an experiment.

    Args:
        name (:py:class:`str`, optional): Name of the experiment (Default: None).
        parameter_grid (:py:class:`dict`, optional): Parameter grid (Default: None).
        parent (:py:class:`~experitur.Experiment`, optional): Parent experiment (Default: None).
        meta (:py:class:`dict`, optional): Dict with experiment metadata that should be recorded.
        active (:py:class:`bool`, optional): Is the experiment active? (Default: True).
            When False, the experiment will not be executed.
        volatile (:py:class:`bool`, optional): If True, the results of a successful run will not be saved (Default: False).
        minimize (:py:class:`str` or list of str, optional): Metric or list of metrics to minimize.
        maximize (:py:class:`str` or list of str, optional): Metric or list of metrics to maximize.
        doc (str, optional): Docstring for this experiment.
        defaults (dict, optional): Default values for parameters.

    This can be used as a constructor or a decorator:

    .. code-block:: python

        # When using as a decorator, the name of the experiment is automatically inferred.
        @Experiment(...)
        def exp1(trial):
            ...

        # Here, the name must be supplied.
        exp2 = Experiment("exp2", parent=exp1)

    When the experiment is run, `trial` will be a :py:class:`~experitur.Trial` instance.
    As such, it has the following characteristics:

    - :obj:`dict`-like interface (`trial[<name>]`): Get the value of the parameter named `name`.
    - Attribute interface (`trial.<attr>`): Get meta-data for this trial.
    - :py:meth:`~experitur.Trial.call`: Run a function and automatically assign parameters.

    See :py:class:`~experitur.Trial` for more details.
    """

    class Optimize(Enum):
        MAXIMIZE = "max"
        MINIMIZE = "min"

    def __init__(
        self,
        name: Optional[str] = None,
        parameters=None,
        configurator=None,
        parent: "Experiment" = None,
        meta: Optional[Mapping] = None,
        active: bool = True,
        volatile: Optional[bool] = None,
        minimize: Union[str, List[str], None] = None,
        maximize: Union[str, List[str], None] = None,
        depends_on: Union[None, "Experiment", List["Experiment"]] = None,
        doc: Optional[str] = None,
        defaults=None,
    ):
        super().__init__()

        if not (isinstance(name, str) or name is None):
            raise ValueError(f"'name' has to be a string or None, got {name!r}")

        self.ctx = get_current_context()
        self.name = name
        self.parent = parent

        # Local import to avoid circular import error
        from experitur import __version__

        self.meta = merge_dicts(
            {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "experitur_version": __version__,
            },
            meta,
        )
        self.active = active
        self.volatile = volatile

        self.optimize: Dict[
            str, Experiment.Optimize
        ] = self._validate_minimize_maximize(minimize, maximize)

        self.depends_on: List["Experiment"] = (
            [depends_on]
            if isinstance(depends_on, Experiment)
            else []
            if depends_on is None
            else depends_on
        )

        # Deprecation of parameters
        if parameters is not None:
            warnings.warn(
                "parameters is deprecated. Use configurator instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if configurator is not None:
                raise ValueError(
                    "parameters and configurator can not be set at the same time."
                )

            configurator = parameters

        self._own_configurators: List[BaseConfigurator] = validate_configurators(
            configurator
        )

        self._pre_trial = None
        self._commands: Dict[str, Any] = {}

        self.func: Optional[Callable[[Trial], Any]] = None

        if defaults is None:
            defaults = {}
        self.defaults = defaults

        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Merge settings from all ancestors
        self._merge_parents()

        # If volatile was not set by any parent, set now.
        if self.volatile is None:
            self.volatile = False

        self._base_configurators: List[BaseConfigurator] = (
            [] if self.parent is None else self.parent._configurators
        )

        self.ctx._register_experiment(self)

    def _merge_parents(self):
        """
        Merge configuration of all ancestors into self.

        This does not include configurators!
        """

        parent: "Optional[Experiment]" = self.parent
        while parent is not None:
            # Copy attributes: func, meta, ...
            for name in ("func", "meta", "defaults"):
                ours = getattr(self, name)
                theirs = getattr(parent, name)

                if ours is None and theirs is not None:
                    # Shallow-copy regular attributes
                    setattr(self, name, copy.copy(theirs))
                elif isinstance(ours, dict) and isinstance(theirs, dict):
                    # Merge dict attributes
                    setattr(self, name, {**theirs, **ours})

            self._event_handlers.update(parent._event_handlers)

            parent = parent.parent

    @staticmethod
    def _validate_minimize_maximize(
        minimize: Union[str, List[str], None], maximize: Union[str, List[str], None]
    ) -> Tuple[List[str], List[str]]:
        minimize, maximize = ensure_list(minimize), ensure_list(maximize)

        common = set(minimize) & set(maximize)

        if common:
            common = ", ".join(sorted(common))
            raise ValueError(f"minimize and maximize share common metrics: {common}")

        optimize = {}
        for m in maximize:
            optimize[m] = Experiment.Optimize.MAXIMIZE
        for m in minimize:
            optimize[m] = Experiment.Optimize.MAXIMIZE

        return optimize

    @property
    def maximize(self):
        return sorted(
            k for k, v in self.optimize.items() if v == Experiment.Optimize.MAXIMIZE
        )

    @property
    def minimize(self):
        return sorted(
            k for k, v in self.optimize.items() if v == Experiment.Optimize.MINIMIZE
        )

    def add_dependency(self, dependency: "Experiment"):
        self.depends_on.append(dependency)

    def __call__(self, func: Callable) -> "Experiment":
        """
        Register an entry-point.

        Allows an Experiment object to be used as a decorator::

            @Experiment()
            def entry_point(trial):
                ...
        """

        if not self.name:
            self.name = func.__name__

        self.func = func

        return self

    def _register_handler(self, event: str, func: Callable):
        self._event_handlers[event].append(func)

    def _handle_event(self, event: str, *args):
        for handler in self._event_handlers[event]:
            handler(*args)

    def on_pre_run(self, func: Callable[[Trial], Any]):
        """
        Register a callback that is called before a trial runs.

        Example:
            experiment = Experiment(...)
            @experiment.on_pre_run
            def on_experiment_pre_run(trial: Trial):
                ...
        """

        self._register_handler("on_pre_run", func)

    def on_success(self, func: Callable[[Trial], Any]):
        """
        Register a callback that is called after a trial finished successfully.

        Example:
            experiment = Experiment(...)
            @experiment.on_success
            def on_experiment_success(trial: Trial):
                ...
        """

        self._register_handler("on_success", func)

    def on_update(self, func: Callable[[Trial], Any]):
        """
        Register a callback that is called when a trial is updated.

        Example:
            experiment = Experiment(...)
            @experiment.on_update
            def on_experiment_update(trial: Trial):
                ...
        """

        self._register_handler("on_update", func)

    def child(
        self,
        name: Optional[str] = None,
        configurator=None,
    ) -> "Experiment":
        """Create a child experiment.

        Args:
            name (str, optional): Name for the child experiment.
                Defaults to the name of the parent.
        """
        if name is None:
            name = self.name

        exp = Experiment(name=name, configurator=configurator, parent=self)

        return exp

    def prepend_configurator(self, configurator: BaseConfigurator) -> None:
        self._own_configurators.insert(0, configurator)

    @property
    def _configurators(self) -> List[BaseConfigurator]:
        return self._base_configurators + self._own_configurators

    @property
    def configurator(self) -> BaseConfigurator:
        return MultiplicativeConfiguratorChain(*self._configurators)

    @property
    def parameters(self) -> List[str]:
        """Names of the configured parameters."""
        return sorted(self.configurator.parameter_values.keys())

    @property
    def varying_parameters(self) -> List[str]:
        """Names of varying parameters, i.e. parameters that can assume more than one value."""
        return sorted(
            k
            for k, v in self.configurator.parameter_values.items()
            if not is_invariant(v)
        )

    @property
    def invariant_parameters(self) -> List[str]:
        """Names of invariant parameters, i.e. parameters that assume only one single value."""
        return sorted(
            k for k, v in self.configurator.parameter_values.items() if is_invariant(v)
        )

    def __str__(self):
        if self.name is not None:
            return self.name
        return repr(self)

    def __repr__(self):  # pragma: no cover
        return "Experiment(name={})".format(self.name)

    def _trial_generator(self):
        """Yields readily created trials."""
        # Generate trial configurations

        sampler = self.configurator.build_sampler()

        skip_cache = _SkipCache(self)

        for trial_configuration in sampler:
            # Inject experiment data into trial_configuration
            # TODO: Insert runtime meta elsewhere
            trial_configuration = self._setup_trial_configuration(trial_configuration)

            # Remove "unset" parameters
            parameters = trial_configuration["parameters"] = clean_unset(
                trial_configuration.get("parameters", {})
            )

            # Run the pre-trial hook to allow the user to interact
            # with the parameters before the trial is created and run.
            if self._pre_trial is not None:
                self._pre_trial(self.ctx, trial_configuration)

            if self.ctx.config["skip_existing"]:
                existing_trials = skip_cache.get_trials(parameters)
                if existing_trials:
                    cprint(
                        f"Skipping existing configuration ({format_trial_parameters(func=self.func, parameters=parameters)}): "
                        + (", ".join(t.id for t in existing_trials)),
                        color="white",
                        attrs=["dark"],
                    )

                    continue

            trial_configuration = merge_dicts(
                trial_configuration,
                resolved_parameters=RecursiveDict(
                    trial_configuration["parameters"], allow_missing=True
                ).as_dict(),
            )

            trial = self.ctx.trials.create(trial_configuration)
            os.makedirs(trial.wdir, exist_ok=True)

            yield trial

    def run(self):
        """
        Run this experiment.

        Create trials for every combination in the parameter grid and run them.
        """

        if not self.active:
            print("Skip inactive experiment {}.".format(self.name))
            return

        if self.func is None:
            raise ValueError("No function was registered for {}.".format(self))

        if self.name is None:
            raise ValueError("Experiment has no name {}.".format(self))

        print("Experiment", self)

        # Print varying parameters of this experiment
        print("Varying parameters:")
        for k, v in sorted(self.configurator.parameter_values.items()):
            if is_invariant(v):
                continue

            print("{}: {}".format(k, v))
        print()

        trials = self._trial_generator()

        if self.ctx.config["resume_failed"]:
            hostname = self.meta.get("hostname", object())

            # TODO: Make sure that these are not resumed simultaneously by another process!
            failed_with_checkpoint = (
                self.trials.filter(
                    lambda trial: trial.is_failed
                    or trial.is_zombie
                    and trial.is_resumable
                )
                # XXX: Resume only trials that were started on the same host
                # .filter(
                #     lambda trial: trial.get("experiment", {})
                #     .get("meta", {})
                #     .get("hostname", object())
                #     == hostname
                # )
            )

            for t in failed_with_checkpoint:
                print(t.experiment)

            cprint(f"{len(failed_with_checkpoint)} resumable trials")

            def _check_unmodified(t: Trial):
                try:
                    t.save()
                except ModifiedError:
                    return False
                return True

            failed_with_checkpoint = (
                t for t in failed_with_checkpoint if _check_unmodified(t)
            )

            trials = itertools.chain(failed_with_checkpoint, trials)

        for trial in trials:
            print()
            cprint(f"Running trial {trial.id} ...", color="white", attrs=["dark"])

            # Apply defaults
            trial.setdefaults(self.defaults)

            # Run the trial
            try:
                self.run_trial(trial)
            except SkipTrial:
                cprint(
                    "Trial was skipped.",
                    color="red",
                    attrs=["dark"],
                )
                trial.remove()
            except Exception:  # pylint: disable=broad-except
                if not self.ctx.config["catch_exceptions"]:
                    raise
            else:
                if self.volatile:
                    trial.remove()
            finally:
                self.ctx.on_trial_end()

            try:
                self.ctx.check_stop()
            except ExperimentStoppedError:
                cprint(
                    "Execution of further trials stopped by signal.",
                    color="red",
                    attrs=["dark"],
                )
                raise

    def run_trial(self, trial: Trial):
        """Run the current trial and save the results."""

        assert self.func is not None

        if trial.is_successful:
            raise ValueError(f"Trial {trial.id} was already successful")

        # Reset any errors and end time
        trial.error = None
        trial.time_end = None

        checkpoint = trial.load_checkpoint()

        if checkpoint is None:
            # Record intital state
            trial.success = False
            trial.time_start = datetime.datetime.now()
            trial.result = None

            args = ()
            kwargs = {}
        else:
            args = checkpoint["args"]
            kwargs = checkpoint["kwargs"]

            print(f"Restored checkpoint {trial.checkpoint_fn}")

        # TODO: Properly inject runtime metadata.
        # Otherwise, resumed trials appear as zombies.
        trial.experiment["meta"].update({k: self.meta[k] for k in ("hostname", "pid")})

        trial.save()

        self._handle_event("on_pre_run", trial)

        try:
            with self.ctx.set_current_trial(trial), trial.record_used_parameters():
                result = self.func(trial, *args, **kwargs)

            # Merge returned result into existing result
            trial.result = {**ensure_dict(trial.result), **ensure_dict(result)}

            # Validate result
            self._validate_trial_result(trial.result)

        except (Exception, KeyboardInterrupt) as exc:
            # TODO: Store.log_error()
            # Log complete exc to file
            error_fn = os.path.join(trial.wdir, "error.txt")
            with open(error_fn, "w") as f:
                f.write(str(exc))
                f.write("\n")
                # f.write(traceback.format_exc())
                tb_exc = traceback.TracebackException.from_exception(
                    exc,
                    capture_locals=self.ctx.config["traceback_capture_locals"],
                )
                f.write("".join(tb_exc.format()))
                # f.write("\n")
                # for k, v in inspect.trace()[-1][0].f_locals.items():
                #     f.write(f"{k}: {try_str(v)}\n")

            if _detect_keyboard_interrupt(exc):
                exc = KeyboardInterrupt()

            trial.error = ": ".join(
                filter(None, (exc.__class__.__name__, str(exc).split("\n")[0]))
            )

            print("\n", flush=True)
            cprint(
                f"{trial.error}\n"
                f"{trial.id} failed.\n"
                f"See {error_fn} for the complete traceback.",
                flush=True,
                color="red",
                attrs=["bold"],
            )

            if self.ctx.config["pm"] and not isinstance(
                exc, (KeyboardInterrupt, SkipTrial, ExperimentStoppedError)
            ):
                pdb.post_mortem()
                # Prevent all further trials from running
                self.ctx.stop()

            raise exc
        else:
            print()
            cprint(
                f"{trial.id} succeeded.",
                color="green",
                attrs=["dark"],
            )

            trial.success = True

            if trial.unused_parameters:
                unused_parameters = ", ".join(trial.unused_parameters)
                cprint(
                    f"Some parameters were not used during the execution of the trial: {unused_parameters}",
                    color="red",
                    attrs=["dark"],
                )
        finally:
            trial.time_end = datetime.datetime.now()
            trial.save()

        self._handle_event("on_success", trial)

        return trial.result

    def _setup_trial_configuration(self, trial_configuration):
        trial_configuration.setdefault("parameters", {})
        trial_configuration.setdefault("tags", [])
        return merge_dicts(
            trial_configuration,
            experiment={
                "name": self.name,
                "parent": self.parent.name if self.parent is not None else None,
                "func": callable_to_name(self.func),
                "meta": self.meta,
                # Names of parameters that where actually configured.
                "independent_parameters": self.parameters,
                "varying_parameters": self.varying_parameters,
                "minimize": self.minimize,
                "maximize": self.maximize,
            },
        )

    def _validate_trial_result(self, trial_result: Optional[dict]):
        if trial_result is None:
            trial_result = {}

        if not isinstance(trial_result, dict):
            raise ExperimentError(
                f"Experiments are expected to return a dict, got {trial_result!r}"
            )

        missing_metrics = (
            set(self.maximize) | set(self.minimize)
        ) - trial_result.keys()

        if missing_metrics:
            missing_metrics = ", ".join(sorted(missing_metrics))
            raise ExperimentError(
                f"Missing metrics in result: {missing_metrics}.\n"
                f"Available metrics: {sorted(trial_result.keys())}"
            )

        return trial_result

    def update(self):
        """
        Run on_update hook for all existing trials.
        """

        for trial in self.trials.filter(lambda trial: trial.is_successful):
            orig_data = copy.deepcopy(trial._data)
            self._handle_event("on_update", trial)

            yield orig_data, trial

    def pre_trial(self, func):
        """Update the pre-trial hook.

        The pre-trial hook is called after the parameters for a trial are
        calculated and before its ID is calculated and it is run.
        This hook can be used to alter the parameters.

        Use :code:`pre_trial(None)` to reset the hook.

        This can be used as a decorator::

            @experiment()
            def exp(trial):
                ...

            @exp.pre_trial
            def pre_trial_handler(ctx, trial_parameters):
                ...

        Args:
            func: A function with the signature (ctx, trial_parameters).
        """

        self._pre_trial = func

    def command(self, name=None, *, target="trial"):
        """Attach a command to an experiment.

        .. code-block:: python

            @experiment()
            def experiment1(trial):
                ...

            @experiment1.command()
            def frobnicate(trial):
                ...

        """

        if target not in ("trial", "experiment"):
            msg = "target has to be one of 'trial', 'experiment', not {}.".format(
                target
            )
            raise ValueError(msg)

        def _decorator(f):
            _name = name or f.__name__

            self._commands[_name] = (f, target)

            return f

        return _decorator

    def do(self, cmd_name, target_name, cmd_args):
        try:
            cmd, target = self._commands[cmd_name]
        except KeyError:
            raise CommandNotFoundError(cmd_name)

        if target == "trial":
            try:
                trial = self.ctx.store[target_name]
            except KeyError as exc:
                raise TrialNotFoundError(target_name) from exc

            # Inject the Trial
            cmd_wrapped = functools.partial(cmd, Trial(trial, self.ctx.trials))
            # Copy over __click_params__ if they exist
            try:
                cmd_wrapped.__click_params__ = cmd.__click_params__  # type: ignore
            except AttributeError:
                pass

            cmd = click.command(name=cmd_name)(cmd_wrapped)
            cmd.main(args=cmd_args, standalone_mode=False)
        elif target == "experiment":
            # Inject self
            cmd_wrapped = functools.partial(cmd, self)
            # Copy over __click_params__ if they exist
            try:
                cmd_wrapped.__click_params__ = cmd.__click_params__  # type: ignore
            except AttributeError:
                pass

            cmd = click.command(name=cmd_name)(cmd_wrapped)
            cmd.main(args=cmd_args, standalone_mode=False)
        else:
            msg = "target={} is not implemented.".format(target)
            raise NotImplementedError(msg)

    @property
    def trials(self):
        if self.name is not None:
            return self.ctx.trials.match(experiment=self)

        raise ValueError(
            f"Experiment has no name set. Set a name explicitely if you intend to use Experiment.trials in the DOX."
        )

    def get_matching_trials(self, exclude=None):
        """
        Return all trials that match the configuration of this experiment.

        This includes all trials with matching configuration, regardless whether they belong to this or to another experiment.
        """

        parameters = self.parameters

        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]

            parameters = [k for k in parameters if k not in exclude]

        sampler = self.configurator.build_sampler()

        return self.ctx.trials.match(func=self.func).filter(
            lambda trial: sampler.contains_subset_of({"parameters": dict(trial)})
        )
