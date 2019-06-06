import collections
import functools
import os
import pprint
import random
from itertools import product

import tqdm

from experitur.helpers.merge_dicts import merge_dicts


class TrialAttribute:
    def __init__(self, value):
        self._value = value

    def __str__(self):
        """
        Get value of this current trial.
        """
        return self._value

    def __getattr__(self, name):
        """
        Look up value in the list of existing trials.
        """
        if name == "_parent":
            # TODO: Get attribute value of corresponding trial of parent
            raise NotImplementedError()
        raise NotImplementedError()


class Trial:
    def __init__(self, experiment, id, parameters):
        self.experiment = experiment
        self.parameters = parameters
        self.id = id
        self.result = None
        self.wdir = os.path.join(self.experiment.ctx.wdir, self.id)

        os.makedirs(self.wdir, exist_ok=True)

    def run(self):
        self.result = self.experiment.callable(self)

        return self.result

    def get_trial_dict(self):
        clbl = self.experiment.callable
        trial_dict = {
            "id": self.id,
            "callable": "{}.{}".format(clbl.__module__, clbl.__name__),
            "parameters": self.parameters,
            "result": self.result,
        }

        return trial_dict

    # see experitur.util
    def register_defaults(self):
        ...

    def apply(self):
        ...


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


class Experiment:
    """
    Experiment decorator.
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
        trials = list(parameter_product(self.parameter_grid))

        if self.ctx.shuffle_trials:
            print("Trials are shuffled.")
            random.shuffle(trials)

        pbar = tqdm.tqdm(total=len(trials), unit="")
        for trial in trials:
            # Check, if a trial with this parameter set already exists
            # TODO: Check callable name
            existing = self.ctx.backend.find_trials_by_parameters(
                self.name, trial)

            if self.ctx.skip_existing and len(existing):
                print("Skip existing trial: {}".format(
                    self.ctx.format_independent_parameters(trial, independent_parameters)))
                continue

            trial_id = self.ctx.backend.make_trial_id(
                trial, independent_parameters)

            trial = Trial(self, trial_id, trial)

            pbar.set_description(
                "Trial {}".format(trial.id), refresh=True)
            pbar.update()

            trial.run()

            self.ctx.backend.save_trial(trial)

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
