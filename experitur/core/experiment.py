import collections
import copy
import functools
import os
import pprint
import random
import sys
import textwrap
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Union

import click
import tqdm

from experitur.core.context import get_current_context
from experitur.core.parameters import (
    Grid,
    Multi,
    ParameterGenerator,
    check_parameter_generators,
)
from experitur.errors import ExperiturError
from experitur.helpers import tqdm_redirect
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.context import Context


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
        except Exception:
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

    This can be used as a constructor or a decorator:

    .. code-block:: python

        # When using as a decorator, the name of the experiment is automatically inferred.
        @Experiment(...)
        def exp1(parameters):
            ...

        # Here, the name must be supplied.
        exp2 = Experiment("exp2", parent=exp1)

    When the experiment is run, `parameters` will be a :py:class:`~experitur.core.trial.TrialParameters` instance.
    As such, it has the following characteristics:

    - :obj:`dict`-like interface (`trial[<name>]`): Get the value of the parameter named `name`.
    - Attribute interface (`trial.<attr>`): Get meta-data for this trial.
    - :py:meth:`~experitur.core.trial.apply`: Run a function and automatically assign parameters.

    See :py:class:`~experitur.core.trial.TrialParameters` for more details.
    """

    def __init__(
        self, name=None, parameters=None, parent=None, meta=None, active=True,
    ):
        self.ctx = get_current_context()
        self.name = name
        self.parent = parent
        self.meta = meta
        self.active = active

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

        self.ctx._register_experiment(self)

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

    @property
    def _parameter_generators(self) -> List[ParameterGenerator]:
        return self._base_parameter_generators + self._own_parameter_generators

    def add_parameter_generator(
        self, parameter_generator: ParameterGenerator, prepend=False
    ):
        if prepend:
            self._own_parameter_generators.insert(0, parameter_generator)
        else:
            self._own_parameter_generators.append(parameter_generator)

    @property
    def parameter_generator(self) -> ParameterGenerator:
        return Multi(self._parameter_generators)

    @property
    def independent_parameters(self) -> List[str]:
        """Independent parameters (parameters that are actually varied) of this experiment."""
        return sorted(self.parameter_generator.varying_parameters.keys())

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

        parameter_generator = self.parameter_generator

        print("Independent parameters:")
        for k, v in parameter_generator.varying_parameters.items():
            print("{}: {}".format(k, v))

        # Generate trial configurations
        trial_configurations = parameter_generator.generate(self)

        pbar = tqdm.tqdm(trial_configurations, unit="")
        for trial_configuration in pbar:
            # Run the pre-trial hook to allow the user to interact
            # with the parameters before the trial is created and run.
            if self._pre_trial is not None:
                self._pre_trial(self.ctx, trial_configuration)

            if self.ctx.config["skip_existing"]:
                # Check, if a trial with this parameter set already exists
                existing = self.ctx.store.match(
                    func=self.func,
                    parameters=trial_configuration.get("parameters", {}),
                )
                if len(existing):
                    pbar.write(
                        "Skip existing configuration: {}".format(
                            format_trial_parameters(
                                func=self.func, parameters=trial_configuration
                            )
                        )
                    )
                    pbar.set_description("[Skipped]")
                    continue

            trial = self.ctx.store.create(trial_configuration, self)

            pbar.write("Trial {}".format(trial.id))
            pbar.set_description("Running trial {}...".format(trial.id))

            # Run the trial
            try:
                with tqdm_redirect.redirect_stdout():
                    trial.run()
            except Exception:
                msg = textwrap.indent(traceback.format_exc(-1), "    ")
                pbar.write("{} failed!".format(trial.data["id"]))
                pbar.write(msg)
                if not self.ctx.config["catch_exceptions"]:
                    raise

            pbar.set_description("Running trial {}... Done.".format(trial.id))

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

            from experitur.core.trial import TrialParameters

            # Inject the TrialProxy
            cmd_wrapped = functools.partial(cmd, TrialParameters(trial))
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
