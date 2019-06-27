"""
This is an example for an experitur design of experiments (DOX) file.

Run it with:
    
    experitur run example.py
"""
from pprint import pformat

# Most of the time, you'll just need this:
from experitur import experiment

# experiment is a decorator that turns regular python functions
# into experitur experiments.
# Most of your experiments will vary some parameters. Specify these in parameter_grid.


@experiment(
    parameter_grid={
        "a": [1, 2],
        "b": [3, 4],
    })
def experiment1(trial):
    """This is the first experiment."""
    print("I am experiment1!")
    pformat(trial)


# This is the second experiment. Being derived from the first, it uses the same function (experiment1).
# However, it will be called with different parameters.
experiment2 = experiment(
    "experiment2",
    parameter_grid={
        "a": [4, 5],
        "b": [-1, -2]
    },
    parent=experiment1)

# The third experiment uses the same parameters as experiment1 but a different function.


@experiment(
    parent=experiment1
)
def experiment3(trial):
    print("I am experiment3!")

# Parameter substitution
# A core feature of experitur is the parameter substitution. The format strings can even be nested!
# Use this, if you need different settings for each dataset for example:


@experiment(
    parameter_grid={
        "dataset": ["bees", "flowers"],
        "dataset_fn": ["/data/{dataset}/index.csv"],
        "bees-crop": [10],
        "flowers-crop": [0],
        "crop": ["{{dataset}-crop}"]
    }
)
def experiment4(trial):
    print(trial["dataset_fn"], trial["crop"])


@experiment4.post_grid
def postgrid4(ctx, parameters_per_trial):
    # Only run every second trial
    for i, p in enumerate(parameters_per_trial):
        if i % 2 == 0:
            yield p


@experiment(meta={"foo": "bar"})
def experiment5(trial):
    raise NotImplementedError("experiment5 is not implemented.")

@experiment(active=False)
def experiment6(trial):
    raise NotImplementedError("experiment6 is not implemented.")