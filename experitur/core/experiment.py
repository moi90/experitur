import collections
import copy
import functools
import os
import pprint
import random
import sys
import textwrap
import traceback
from typing import TYPE_CHECKING, List, Union

import click
import tqdm

import experitur.core.samplers as _samplers
from experitur.errors import ExperiturError
from experitur.helpers import tqdm_redirect
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict

if TYPE_CHECKING:
    from experitur.core.context import Context

_callable = callable


class ExperimentError(ExperiturError):
    pass


class StopExecution(ExperimentError):
    pass


class CommandNotFoundError(ExperimentError):
    pass


class TrialNotFoundError(ExperimentError):
    pass


def format_trial_parameters(callable=None, parameters=None, experiment=None):
    if callable is not None:
        try:
            callable = callable.__name__
        except:
            callable = str(callable)
    else:
        callable = "_"

    if parameters is not None:
        parameters = (
            "("
            + (", ".join("{}={}".format(k, repr(v)) for k, v in parameters.items()))
            + ")"
        )
    else:
        parameters = "()"

    if experiment is not None:
        callable = "{}:{}".format(str(experiment), callable)

    return callable + parameters


class Experiment:
    """An experiment."""

    def __init__(
        self,
        ctx: "Context",
        name=None,
        sampler: Union[List[_samplers.Sampler], _samplers.Sampler] = None,
        parameter_grid=None,
        parent=None,
        meta=None,
        active=True,
    ):
        self.ctx = ctx
        self.name = name
        self.parent = parent
        self.meta = meta
        self.active = active

        # Legacy parameter_grid
        if parameter_grid is not None:
            if sampler is not None:
                raise ValueError("parameter_grid and samplers are mutually exclusive")
            sampler = _samplers.GridSampler(parameter_grid)

        # Concatenate samplers
        if parent is not None:
            if isinstance(parent.sampler, _samplers.MultiSampler):
                tmp_sampler = copy.copy(parent.sampler)
            else:
                tmp_sampler = _samplers.MultiSampler([self.parent.sampler])

            if isinstance(sampler, list):
                tmp_sampler.addMulti(sampler)
            elif sampler is not None:
                tmp_sampler.add(sampler)

            self.sampler = tmp_sampler
        else:
            if sampler is None:
                sampler = _samplers.GridSampler({})
            self.sampler = sampler

        self._pre_trial = None
        self._update = None
        self._commands = {}

        self.callable = None

        # Merge parameters from all ancestors
        parent = self.parent
        while parent:
            self._merge(parent)
            parent = parent.parent

        self.ctx._register_experiment(self)

    def __call__(self, callable):
        """Register an entry-point.

        Allows an Experiment object to be used as a decorator::

            @Experiment()
            def entry_point(trial):
                ...
        """

        if not self.name:
            self.name = callable.__name__

        self.callable = callable

        return self

    @property
    def independent_parameters(self):
        """Independent parameters (parameters that are actually varied) of this experiment."""
        return sorted(self.sampler.varying_parameters.keys())

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

        if self.callable is None:
            raise ValueError("No callable was registered for {}.".format(self))

        if self.name is None:
            raise ValueError("Experiment has no name {}.".format(self))

        print("Experiment", self)

        print("Independent parameters:")
        for k, v in self.sampler.varying_parameters.items():
            print("{}: {}".format(k, v))

        # Generate trial configurations
        trial_configurations = self.sampler.sample(self)

        pbar = tqdm.tqdm(trial_configurations, unit="")
        for trial_configuration in pbar:
            # Perform parameter substitution

            # Run the pre-trial hook to allow the user to interact
            # with the parameters before the trial is created and run.
            if self._pre_trial is not None:
                self._pre_trial(self.ctx, trial_configuration)

            if self.ctx.config["skip_existing"]:
                # Check, if a trial with this parameter set already exists
                existing = self.ctx.store.match(
                    callable=self.callable,
                    parameters=trial_configuration["parameters"],
                )
                if len(existing):
                    pbar.write(
                        "Skip existing configuration: {}".format(
                            format_trial_parameters(
                                callable=self.callable, parameters=trial_configuration
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

        `other` is usually the parent experiment.
        """

        # Copy attributes: callable, ...
        for name in ("callable", "meta"):
            ours = getattr(self, name)
            theirs = getattr(other, name)

            if ours is None and theirs is not None:
                # Shallow-copy regular attributes
                setattr(self, name, copy.copy(theirs))
            elif isinstance(ours, dict) and isinstance(theirs, dict):
                # Merge dict attributes
                setattr(self, name, {**theirs, **ours})

    def pre_trial(self, callable):
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
            callable: A callable with the signature (ctx, trial_parameters).
        """

        self._pre_trial = callable

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
            raise CommandNotFoundError(cmd)

        if target == "trial":
            try:
                trial = self.ctx.store[target_name]
            except KeyError as exc:
                raise TrialNotFoundError(target_name) from exc

            from experitur.core.trial import TrialProxy

            # Inject the TrialProxy
            cmd_wrapped = functools.partial(cmd, TrialProxy(trial))
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
