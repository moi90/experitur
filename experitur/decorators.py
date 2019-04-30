import collections
import functools
import pprint
import random
from itertools import product
import tqdm

from experitur.helpers.merge_dicts import merge_dicts


"""
Decorators to use in user scripts.
"""


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
    def __init__(self, experiment, parameters):
        self.experiment = experiment
        self.parameters = parameters
        self._id = ...

    def run(self):
        return ...

    @property
    def id(self):
        return self._id


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


def default_trial_generator(experiment):
    """
    This trial generator builds the cross-product for the parameter grid.
    """


class Experiment:
    """
    Experiment decorator.
    """

    def __init__(self, ctx, name=None, parameter_grid=None, parent=None):
        self.ctx = ctx
        self.name = name
        self.parameter_grid = parameter_grid
        self.parent = parent

        self.callable = None

        self.ctx.register_experiment(self)

    def __call__(self, clbl):
        """
        Register a callable.
        """
        if self.callable:
            raise ValueError(
                "A callable was already registered for {}.".format(self))

        if not self.name:
            self.name = clbl.__name__

        self.callable = clbl

        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Experiment(name={})".format(self.name)

    def run(self):
        # Merge parameters from parents
        parent = self.parent
        while parent:
            self.merge(parent)
            parent = parent.parent

        if self.callable is None:
            raise ValueError("No callable was registered for {}.".format(self))
        print("Running", self)

        pprint.pprint(self.parameter_grid)

        # For every point in the parameter grid create a trial
        trials_parameters = list(parameter_product(self.parameter_grid))

        if self.ctx.shuffle_trials:
            print("Trials are shuffled.")
            random.shuffle(trials_parameters)

        pbar = tqdm.tqdm(total=len(trials_parameters), unit="")
        for tpar in trials_parameters:
            # Check, if a trial with this parameter set already exists
            existing = ctx.find_trials_by_parameters(tpar)

            if self.ctx.skip_existing and len(existing):
                print("Skip existing trial: {}".format(
                    ctx.format_independent_parameters(trial_parameters, independent_parameters)))
                continue

            trial = Trial(self, tpar)

            pbar.set_description(
                "Trial {}".format(trial.id), refresh=True)
            pbar.update()

            result = trial.run()

            self.ctx.save_result(result)

    def merge(self, other):
        """
        Merge configuration of other into self.
        """

        for name in ("callable",):
            ours = getattr(self, name)
            theirs = getattr(other, name)

            if ours is None and theirs is not None:
                setattr(self, name, theirs)

        # Merge parameter grid
        self.parameter_grid = merge_dicts(
            self.parameter_grid, other.parameter_grid)
