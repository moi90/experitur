# experitur

[![](https://img.shields.io/pypi/v/experitur.svg?style=flat)](https://pypi.org/project/experitur/) [![Build Status](https://travis-ci.org/moi90/experitur.svg?branch=master)](https://travis-ci.org/moi90/experitur) [![codecov](https://codecov.io/gh/moi90/experitur/branch/master/graph/badge.svg)](https://codecov.io/gh/moi90/experitur) ![](https://img.shields.io/pypi/pyversions/experitur.svg?style=flat) [![Documentation Status](https://readthedocs.org/projects/experitur/badge/?version=latest)](https://experitur.readthedocs.io/en/latest/?badge=latest)

**experitur** automates machine learning and other computer science experiments and stores the results in an easily accessible format.
It includes grid search, random search, parameter substitution, inheritance and resuming aborted experiments.

![example](https://raw.githubusercontent.com/moi90/experitur/master/examples/simple.gif)

Read the [documentation](https://experitur.readthedocs.io/en/latest/)!

## Experiment description

Every experiment is described in a regular python file. The `@Experiment` decorator is used to mark experiment entry-points.
By default, parameters are defined as a *parameter grid* where each parameter has a list of values that it can take. A number of *trials* is generated from the cross product of the values of each parameter.
(So it works like [`sklearn.model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html).)

An experiment is a regular function that is decorated with `@Experiment` (unless it is *abstract* or *derived*). Upon execution, this function gets called with the current trial's parameters. It may return a result dictionary.

**Signature:** `(trial: Trial) -> Optional[dict]`

```python
from experitur import Experiment, Trial

@Experiment(
    parameters={
        "parameter_1": [1,2,3],
        "parameter_2": ["a", "b", "c"],
    })
def example_experiment(trial: Trial):
    """This is an example experiment."""
    print("parameter_1:", trial["parameter_1"])
    print("parameter_2:", trial["parameter_2"])
    return {}
```

You can run the experiment using `experitur run example.py` and `example_experiment` will be called six times with every combination of [1,2] x [a,b,c].

### Multiple experiments

The Python file can contain multiple experiments:

```python
from experitur import Experiment, Trial

@Experiment(...)
def example1(trial: Trial):
    ...
    
@Experiment(...)
def example2(trial: Trial):
    ...
```

### Experiment inheritance

One experiment may inherit the settings of another, using the `parent` parameter.

```python
from experitur import experiment

@Experiment(...)
def example1(trial):
    ...
    
# Derived with own entry point:
@Experiment(parent=example1)
def example2(trial):
    ...
    
# Derived with inherited entry point:
example3 = experiment("example3", parent=example2)
```

## The `trial` object

Every experiment receives a `Trial` instance that allows access to the parameters and meta-data of the trial.

Parameters are accessed with the `[]` operator (e.g. `trial["a"]`), meta-data is accessed with the `.` operator (e.g. `trial.wdir`).

## Files

When `experitur` executes a script, it creates the following file structure in the directory where the experiment file is located:

```
/
+- script.py
+- script/
|  +- experiment_id/
|  |  +- trial_id/
|  |  |  +- experitur.yaml
|  |  ...
|  ...
```

`<script>/<experiment_id>/<trial_id>/experitur.yaml` contains the parameters and the results from a trial, e.g.:

```yaml
experiment:
  func: simple.simple
  meta: null
  name: simple
  parent: null
id: simple/a-1_b-3
parameters:
  a: 1
  b: 3
result: null
success: true
time_end: 2020-03-26 21:01:51.648282
time_start: 2020-03-26 21:01:51.147210
wdir: simple/simple/a-1_b-3

```

Most items should be self-explanatory. `parameters` are the parameters passed to the entry point. `id` is derived from the parameters that are varied in the parameter grid. This way, you can easily interpret the file structure.

## Installation

**experitur** is packaged on [PyPI](https://pypi.org/project/experitur/).

```shell
pip install experitur
```

Be warned that this package is currently under heavy development and anything might change any time!

To install the development version, do:

```shell
pip install -U git+https://github.com/moi90/experitur.git
```

## Examples

-  [examples/example.py](examples/example.py): A very basic example showing the workings of `set_default_parameters` and `apply_parameters`.
-  [examples/classifier.py](examples/classifier.py): Try different parameters of `sklearn.svm.SVC` to classify handwritten digits (the [MNIST](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) test set). Run the example, add more parameter values and see how `experitur` skips already existing configurations during the next run.

## Contributions
**experitur** is under active development, so any user feedback, bug reports, comments, suggestions, or pull requests are highly appreciated. Please use the [bug tracker](https://github.com/moi90/experitur/issues) and [fork](https://github.com/moi90/experitur/network/members) the repository.

## Compatibility

`experitur` is [tested](https://travis-ci.org/moi90/experitur) with Python 3.6, 3.7 and 3.8.