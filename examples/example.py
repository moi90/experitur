"""
This is an example for an experitur design of experiments (DOX) file.

Run it with:

    experitur run example.py
"""

from pprint import pformat

import click

# Most of the time, you'll just need this:
from experitur import Experiment

# experiment is a decorator that turns regular python functions
# into experitur experiments.
# Most of your experiments will vary some parameters. Specify these in parameter_grid.


@Experiment(configurator={"a": [1, 2], "b": [3, 4]})
def experiment1(trial):
    """This is the first experiment."""
    print("I am experiment1!")
    pformat(trial)


# This is the second experiment. Being derived from the first, it uses the same function (experiment1).
# However, it will be called with different parameters.
experiment2 = Experiment(
    "experiment2", parameters={"a": [4, 5], "b": [-1, -2]}, parent=experiment1
)

# The third experiment uses the same parameters as experiment1 but a different function.


@Experiment(parent=experiment1)
def experiment3(trial):
    print("I am experiment3!")


# Parameter substitution
# A core feature of experitur is the parameter substitution. The format strings can even be nested!
# Use this, if you need different settings for each dataset for example:


@Experiment(
    parameters={
        "dataset": ["bees", "flowers"],
        "dataset_fn": ["/data/{dataset}/index.csv"],
        "bees-crop": [10],
        "flowers-crop": [0],
        "crop": ["{{dataset}-crop}"],
    }
)
def experiment4(trial):
    print(trial["dataset_fn"], trial["crop"])


@Experiment(meta={"foo": "bar"})
def experiment5(trial):
    raise NotImplementedError("experiment5 is not implemented.")


@Experiment(active=False)
def experiment6(trial):
    raise NotImplementedError("experiment6 is not implemented.")


@experiment5.command("test", target="trial")
@click.option("--shout/--no-shout", default=False)
def experiment5_test(trial, shout):
    print(trial)
    print(repr(shout))
