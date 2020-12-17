import copy
import datetime
import functools
import inspect
import os
import socket
import textwrap
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Union

import click
import tqdm

import experitur
from experitur.core.context import get_current_context
from experitur.core.parameters import (
    Multi,
    ParameterGenerator,
    check_parameter_generators,
    count_values,
)
from experitur.core.trial import Trial
from experitur.errors import ExperiturError
from experitur.helpers import tqdm_redirect
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict
from experitur.util import callable_to_name, ensure_dict, ensure_list

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


class _SkipCache:
    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment

        self._trial_collection: Optional[TrialCollection] = None

    def __contains__(self, parameters):
        if self._trial_collection is not None:
            existing = self._trial_collection.match(resolved_parameters=parameters)
            if existing:
                return True

        # If configuration was not yet found, update cache...
        self._trial_collection = self.experiment.ctx.get_trials(
            func=self.experiment.func
        )

        # ... and retry
        existing = self._trial_collection.match(resolved_parameters=parameters)
        if existing:
            return True

        return False


class _TempParameterGeneratorContext:
    def __init__(self, experiment: "Experiment", parameter_generator):
        self.experiment = experiment
        self.parameter_generator = parameter_generator

    def __enter__(self):
        pass

    def __exit__(self, *_, **__):
        self.experiment.remove_parameter_generator(self.parameter_generator)


class Experiment:
    """
    Define an experiment.

    Args:
        name (:py:class:`str`, optional): Name of the experiment (Default: None).
        parameter_grid (:py:class:`dict`, optional): Parameter grid (Default: None).
        parent (:py:class:`~experitur.core.experiment.Experiment`, optional): Parent experiment (Default: None).
        meta (:py:class:`dict`, optional): Dict with experiment metadata that should be recorded.
        active (:py:class:`bool`, optional): Is the experiment active? (Default: True).
            When False, the experiment will not be executed.
        volatile (:py:class:`bool`, optional): If True, the results of a successful run will not be saved (Default: False).
        minimize (str or list of str, optional): Metric or list of metrics to minimize.
        maximize (str or list of str, optional): Metric or list of metrics to maximize.

    This can be used as a constructor or a decorator:

    .. code-block:: python

        # When using as a decorator, the name of the experiment is automatically inferred.
        @Experiment(...)
        def exp1(trial):
            ...

        # Here, the name must be supplied.
        exp2 = Experiment("exp2", parent=exp1)

    When the experiment is run, `trial` will be a :py:class:`~experitur.core.trial.Trial` instance.
    As such, it has the following characteristics:

    - :obj:`dict`-like interface (`trial[<name>]`): Get the value of the parameter named `name`.
    - Attribute interface (`trial.<attr>`): Get meta-data for this trial.
    - :py:meth:`~experitur.core.trial.apply`: Run a function and automatically assign parameters.

    See :py:class:`~experitur.core.trial.Trial` for more details.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        parameters=None,
        parent: "Experiment" = None,
        meta: Optional[Mapping] = None,
        active: bool = True,
        volatile: bool = False,
        minimize: Union[str, List[str], None] = None,
        maximize: Union[str, List[str], None] = None,
        depends_on: Optional[List["Experiment"]] = None,
    ):
        if not (isinstance(name, str) or name is None):
            raise ValueError(f"'name' has to be a string or None, got {name!r}")

        self.ctx = get_current_context()
        self.name = name
        self.parent = parent
        self.meta = merge_dicts(
            {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "experitur_version": experitur.__version__,
            },
            meta,
        )
        self.active = active
        self.volatile = volatile

        self.minimize, self.maximize = self._validate_minimize_maximize(
            minimize, maximize
        )

        self.depends_on: List["Experiment"] = [depends_on] if isinstance(
            depends_on, Experiment
        ) else [] if depends_on is None else depends_on

        self._own_parameter_generators: List[ParameterGenerator]
        self._own_parameter_generators = check_parameter_generators(parameters)

        self._pre_trial = None
        self._commands: Dict[str, Any] = {}

        self.func = None

        # Merge parameters from all ancestors
        parent = self.parent
        while parent is not None:
            self._merge(parent)
            parent = parent.parent

        self._base_parameter_generators: List[ParameterGenerator]
        self._base_parameter_generators = (
            [] if self.parent is None else self.parent._parameter_generators
        )

        self._on_success = []

        self.ctx._register_experiment(self)

    @staticmethod
    def _validate_minimize_maximize(minimize, maximize):
        minimize, maximize = ensure_list(minimize), ensure_list(maximize)

        common = set(minimize) & set(maximize)

        if common:
            common = ", ".join(sorted(common))
            raise ValueError(f"minimize and maximize share common metrics: {common}")

        return minimize, maximize

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

    def on_success(self, func: Callable[[Trial], Any]):
        """
        Register a callback that is called after a trial finished successfully.

        Example:
            experiment = Experiment(...)
            @experiment.on_success
            def on_experiment_success(trial: Trial):
                ...
        """

        self._on_success.append(func)

    def _handle_success(self, trial: Trial):
        for handler in self._on_success:
            handler(trial)

    @property
    def _parameter_generators(self) -> List[ParameterGenerator]:
        return self._base_parameter_generators + self._own_parameter_generators

    def add_parameter_generator(
        self, parameter_generator: ParameterGenerator, prepend=False
    ):
        """
        Add a ParameterGenerator to the Experiment.

        When used as a context manager, the ParameterGenerator is removed when the context is exited.
        """
        if prepend:
            self._own_parameter_generators.insert(0, parameter_generator)
        else:
            self._own_parameter_generators.append(parameter_generator)

        return _TempParameterGeneratorContext(self, parameter_generator)

    def remove_parameter_generator(self, parameter_generator: ParameterGenerator):
        self._own_parameter_generators.remove(parameter_generator)

    @property
    def parameter_generator(self) -> ParameterGenerator:
        return Multi(self._parameter_generators)

    @property
    def independent_parameters(self) -> List[str]:
        """Independent parameters. (Parameters that were actually configured.)"""
        return sorted(self.parameter_generator.independent_parameters.keys())

    @property
    def varying_parameters(self) -> List[str]:
        """Varying parameters of this experiment."""
        return sorted(
            k
            for k, v in self.parameter_generator.independent_parameters.items()
            if count_values(v) != 1
        )

    @property
    def invariant_parameters(self) -> List[str]:
        """Varying parameters of this experiment."""
        return sorted(
            k
            for k, v in self.parameter_generator.independent_parameters.items()
            if count_values(v) == 1
        )

    def __str__(self):
        if self.name is not None:
            return self.name
        return repr(self)

    def __repr__(self):  # pragma: no cover
        return "Experiment(name={})".format(self.name)

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

        print("Varying parameters:")
        for k, v in sorted(self.parameter_generator.independent_parameters.items()):
            if count_values(v) == 1:
                continue
            print("{}: {}".format(k, v))
        print()

        # Generate trial configurations
        trial_configurations = self.parameter_generator.generate(self)

        skip_cache = _SkipCache(self)

        pbar = tqdm.tqdm(trial_configurations, unit="")
        for trial_configuration in pbar:
            # Inject experiment data into trial_configuration
            trial_configuration = self._setup_trial_configuration(trial_configuration)

            # Run the pre-trial hook to allow the user to interact
            # with the parameters before the trial is created and run.
            if self._pre_trial is not None:
                self._pre_trial(self.ctx, trial_configuration)

            parameters = trial_configuration.get("parameters", {})

            if self.ctx.config["skip_existing"] and parameters in skip_cache:
                pbar.write(
                    "Skip existing configuration: {}".format(
                        format_trial_parameters(func=self.func, parameters=parameters)
                    )
                )
                pbar.write("")
                pbar.set_description("[Skipped]")
                continue

            trial_configuration = merge_dicts(
                trial_configuration,
                resolved_parameters=RecursiveDict(
                    trial_configuration["parameters"], allow_missing=True
                ).as_dict(),
            )

            trial = self.ctx.trials.create(trial_configuration)
            os.makedirs(trial.wdir, exist_ok=True)
            trial._record_used_parameters = (
                True  # pylint: disable=record_used_parameters
            )

            pbar.write("Trial {}".format(trial.id))
            pbar.set_description("Running trial {}...".format(trial.id))

            # Run the trial
            try:
                with tqdm_redirect.redirect_stdout():
                    result = self.run_trial(trial)
                result = self._validate_trial_result(result)
            except Exception:  # pylint: disable=broad-except
                msg = textwrap.indent(traceback.format_exc(-1), "    ")
                pbar.write("{} failed!".format(trial.id))
                pbar.write(msg)
                if not self.ctx.config["catch_exceptions"]:
                    raise
            else:
                if self.volatile:
                    trial.remove()

            pbar.set_description("Running trial {}... Done.".format(trial.id))

            if self.ctx.should_stop():
                return

    def run_trial(self, trial: Trial):
        """Run the current trial and save the results."""

        # Record intital state
        trial.success = False
        trial.time_start = datetime.datetime.now()
        trial.result = None
        trial.error = None

        trial.save()

        try:
            with self.ctx.set_current_trial(trial):
                result = self.func(trial)
        except (Exception, KeyboardInterrupt) as exc:
            # TODO: Store.log_error()
            # Log complete exc to file
            error_fn = os.path.join(trial.wdir, "error.txt")
            with open(error_fn, "w") as f:
                f.write(str(exc))
                f.write(traceback.format_exc())
                f.write("\n")
                for k, v in inspect.trace()[-1][0].f_locals.items():
                    f.write(f"{k}: {try_str(v)}\n")

            trial.error = ": ".join(filter(None, (exc.__class__.__name__, str(exc))))

            print("\n", flush=True)
            print(
                f"Error running {trial.id}.\n"
                f"See {error_fn} for the complete traceback.",
                flush=True,
            )

            raise exc

        else:
            trial.result = result
            trial.success = True
        finally:
            trial.time_end = datetime.datetime.now()
            trial.save()

        self._handle_success(trial)

        return trial.result

    def _setup_trial_configuration(self, trial_configuration):
        trial_configuration.setdefault("parameters", {})
        return merge_dicts(
            trial_configuration,
            experiment={
                "name": self.name,
                "parent": self.parent.name if self.parent is not None else None,
                "func": callable_to_name(self.func),
                "meta": self.meta,
                # Parameters that where actually configured.
                "independent_parameters": self.independent_parameters,
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
            set(self.maximize) | set(self.maximize)
        ) - trial_result.keys()

        if missing_metrics:
            missing_metrics = ", ".join(sorted(missing_metrics))
            raise ExperimentError(f"Missing metrics in result: {missing_metrics}")

        return trial_result

    def _merge(self, other):
        """
        Merge configuration of other into self.

        This does not include parameter generators!

        `other` is usually the parent experiment.
        """

        # Copy attributes: func, meta, ...
        for name in ("func", "meta"):
            ours = getattr(self, name)
            theirs = getattr(other, name)

            if ours is None and theirs is not None:
                # Shallow-copy regular attributes
                setattr(self, name, copy.copy(theirs))
            elif isinstance(ours, dict) and isinstance(theirs, dict):
                # Merge dict attributes
                setattr(self, name, {**theirs, **ours})

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
            cmd_wrapped = functools.partial(cmd, Trial(trial, self.ctx.store))
            # Copy over __click_params__ if they exist
            try:
                cmd_wrapped.__click_params__ = cmd.__click_params__
            except AttributeError:
                pass

            cmd = click.command(name=cmd_name)(cmd_wrapped)
            cmd.main(args=cmd_args, standalone_mode=False)
        elif target == "experiment":
            # Inject self
            cmd_wrapped = functools.partial(cmd, self)
            # Copy over __click_params__ if they exist
            try:
                cmd_wrapped.__click_params__ = cmd.__click_params__
            except AttributeError:
                pass

            cmd = click.command(name=cmd_name)(cmd_wrapped)
            cmd.main(args=cmd_args, standalone_mode=False)
        else:
            msg = "target={} is not implemented.".format(target)
            raise NotImplementedError(msg)

    @property
    def trials(self):
        return self.ctx.get_trials(experiment=self)
