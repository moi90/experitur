import collections
import functools
import os
import pprint
import random
from itertools import product

import tqdm

from experitur.errors import ExperiturError
from experitur.helpers.merge_dicts import merge_dicts

_callable = callable


class ExperimentError(ExperiturError):
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
    """
    """

    def __init__(self, ctx, name=None, parameter_grid=None, parent=None):
        self.ctx = ctx
        self.name = name
        self.parameter_grid = parameter_grid or {}
        self.parent = parent

        self.callable = None

        # Merge parameters from parents
        parent = self.parent
        while parent:
            self.merge(parent)
            parent = parent.parent

        self.ctx._register_experiment(self)

    def __call__(self, clbl):
        """
        Register a callable.
        Allows an Experiment object to be used as a decorator.
        """

        if not self.name:
            self.name = clbl.__name__

        self.callable = clbl

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

    def __repr__(self):
        return "Experiment(name={})".format(self.name)

    def run(self):
        print("Experiment.run")

        if self.callable is None:
            raise ValueError("No callable was registered for {}.".format(self))

        if self.name is None:
            raise ValueError("Experiment has no name {}.".format(self))

        print("Experiment", self)

        # Select independent parameters
        independent_parameters = self.independent_parameters

        print("Independent parameters:")
        for k in independent_parameters:
            print("{}: {}".format(k, self.parameter_grid[k]))

        # For every point in the parameter grid create a trial
        parameters_per_trial = list(parameter_product(self.parameter_grid))

        if self.ctx.shuffle_trials:
            print("Trials are shuffled.")
            random.shuffle(parameters_per_trial)

        pbar = tqdm.tqdm(total=len(parameters_per_trial), unit="")
        for trial_parameters in parameters_per_trial:
            # Check, if a trial with this parameter set already exists
            # TODO: Check callable name
            existing = self.ctx.store.match(
                callable=self.callable, parameters=trial_parameters)

            if self.ctx.skip_existing and len(existing):
                print("Skip existing configuration: {}".format(format_trial_parameters(
                    callable=self.callable, parameters=trial_parameters)))
                continue

            trial = self.ctx.store.create(trial_parameters, self)

            pbar.set_description(
                "Trial {}".format(trial.id), refresh=True)
            pbar.update()

            trial.run()
            trial.save()

    def merge(self, other):
        """
        Merge configuration of other into self.
        """

        # Copy attributes: callable, ...
        for name in ("callable",):
            ours = getattr(self, name)
            theirs = getattr(other, name)

            if ours is None and theirs is not None:
                setattr(self, name, theirs)

        if self.parameter_grid is None:
            self.parameter_grid = other.parameter_grid
            return

        # Merge parameter grid of other (and its parents)
        self.parameter_grid = merge_dicts(
            self.parameter_grid, other.parameter_grid)
