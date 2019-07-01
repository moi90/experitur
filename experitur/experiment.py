import collections
import copy
import functools
import os
import pprint
import random
import sys
import textwrap
import traceback
from itertools import product

import tqdm

from experitur.errors import ExperiturError
from experitur.helpers import tqdm_redirect
from experitur.recursive_formatter import RecursiveDict
from experitur.trial import TrialProxy

_callable = callable


class ExperimentError(ExperiturError):
    pass


class StopExecution(ExperimentError):
    pass


def parameter_product(p):
    """Iterate over the points in the grid."""

    items = sorted(p.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params


def format_trial_parameters(callable=None, parameters=None, experiment=None):
    if callable is not None:
        try:
            callable = callable.__name__
        except:
            callable = str(callable)
    else:
        callable = '_'

    if parameters is not None:
        parameters = '(' + (', '.join("{}={}".format(k, repr(v))
                                      for k, v in parameters.items())) + ')'
    else:
        parameters = '()'

    if experiment is not None:
        callable = '{}:{}'.format(str(experiment), callable)

    return callable + parameters


class Experiment:
    """An experiment.
    """

    def __init__(self, ctx, name=None, parameter_grid=None, parent=None, meta=None, active=True):
        self.ctx = ctx
        self.name = name
        self.parameter_grid = {} if parameter_grid is None else parameter_grid
        self.parent = parent
        self.meta = meta
        self.active = active
        self._pre_trial = None
        self._post_grid = None
        self._update = None

        self.callable = None

        # Merge parameters from parents
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
        """
        Calculate independent parameters (parameters that are actually varied) of this experiment.
        """
        return sorted(
            k for k, v in self.parameter_grid.items() if len(v) > 1)

    def __str__(self):
        return self.name

    def __repr__(self):  # pragma: no cover
        return "Experiment(name={})".format(self.name)

    def _check_parameter_grid(self):
        """
        self.parameter_grid has to be a dict of lists.
        """

        if not isinstance(self.parameter_grid, dict):
            raise ExperimentError(
                "parameter_grid is expected to be a dictionary.")

        errors = []
        for k, v in self.parameter_grid.items():
            if not isinstance(v, list):
                errors.append(k)

        if errors:
            raise ExperimentError(
                "Parameters {} are not lists.".format(", ".join(errors)))

    def run(self):
        """Runs this experiment.

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

        self._check_parameter_grid()

        # Select independent parameters
        independent_parameters = self.independent_parameters

        print("Independent parameters:")
        for k in independent_parameters:
            print("{}: {}".format(k, self.parameter_grid[k]))

        # For every point in the parameter grid calculate the parameters
        # and perform parameter substitution.
        parameters_per_trial = [
            RecursiveDict(p, allow_missing=True).as_dict()
            for p in parameter_product(self.parameter_grid)
        ]

        if self.ctx.config["shuffle_trials"]:
            print("Trials are shuffled.")
            random.shuffle(parameters_per_trial)

        # Apply post-grid hook to allow the user to interact with the generated
        # parameter combinations.

        if self._post_grid is not None:
            parameters_per_trial = self._post_grid(
                self.ctx, parameters_per_trial)

        pbar = tqdm.tqdm(parameters_per_trial, unit="")
        for trial_parameters in pbar:
            # Run the pre-trial hook to allow the user to interact
            # with the parameters before the trial is created and run.
            if self._pre_trial is not None:
                self._pre_trial(self.ctx, trial_parameters)

            # Check, if a trial with this parameter set already exists
            existing = self.ctx.store.match(
                callable=self.callable, parameters=trial_parameters)

            if self.ctx.config["skip_existing"] and len(existing):
                pbar.write("Skip existing configuration: {}".format(format_trial_parameters(
                    callable=self.callable, parameters=trial_parameters)))
                pbar.set_description("[Skipped]")
                continue

            trial = self.ctx.store.create(trial_parameters, self)

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
        """

        # Copy attributes: callable, ...
        for name in ("callable", "parameter_grid", "meta"):
            ours = getattr(self, name)
            theirs = getattr(other, name)

            if ours is None and theirs is not None:
                # Copy regular attributes
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

    def post_grid(self, callable):
        """Update the post-grid hook.

        The post-grid hook is called after the parameter grid has been
        calculated and before every cell of the grid is executed.
        This hook can be used to alter the parameter grid (add/remove/change cells).

        Use :code:`post_grid(None)` to reset the hook.

        The callable is expected to return an iterable of parameter dicts, one for each trial.

        This can be used as a decorator::

            @experiment()
            def exp(trial):
                ...

            @exp.post_grid
            def post_grid_handler(ctx, parameters_per_trial):
                ...
                return parameters_per_trial

        Args:
            callable: A callable with the signature  (ctx, parameters_per_trial) -> parameters_per_trial.
        """

        self._post_grid = callable

    def set_update(self, callable):
        self._update = callable

    def update(self):
        if self._update is None:
            return

        trials = self.ctx.store.match(experiment=self.name)

        pbar = tqdm.tqdm(trials.items(), unit="")

        for trial_id, trial in pbar:
            self._update(TrialProxy(trial))
            trial.save()
