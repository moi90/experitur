# experitur

[![](https://img.shields.io/pypi/v/experitur.svg?style=flat)](https://pypi.org/project/experitur/) [![Build Status](https://travis-ci.org/moi90/experitur.svg?branch=master)](https://travis-ci.org/moi90/experitur) [![codecov](https://codecov.io/gh/moi90/experitur/branch/master/graph/badge.svg)](https://codecov.io/gh/moi90/experitur) ![](https://img.shields.io/pypi/pyversions/experitur.svg?style=flat) [![Documentation Status](https://readthedocs.org/projects/experitur/badge/?version=latest)](https://experitur.readthedocs.io/en/latest/?badge=latest)

Automates machine learning and other computer experiments. Includes grid search and resuming aborted experiments. No lock-in, all your data is easily accessible in a text-based, machine-readable format.

![example](examples/simple.gif)

## Experiment description

Every experiment is described in a regular python file. The `@experiment` decorator is used to mark experiment entry-points.

```python
from experitur import experiment

@experiment(
    parameter_grid={
        "parameter_1": [1,2,3],
        "parameter_2": ["a", "b", "c"],
    })
def example(trial):
    """This is an example experiment."""
    ...
```

### Parameter grid

The core of an experiment is its *parameter grid*. It works like [`sklearn.model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html). Each parameter has a list of values that it can take. A number of *trials* is generated from the cross product of the values of each parameter.

### Entry point

An experiment is a regular function that is decorated with `@experiment` (unless it is *abstract* or *derived*). Upon execution, the function gets called with the current trial. It may return a result dictionary.

**Signature:** `(trial) -> dict`

```python
from experitur import experiment

@experiment(
    parameter_grid={
        "parameter_1": [1,2,3],
        "parameter_2": ["a", "b", "c"],
    })
def example(trial):
    """This is an example experiment."""
    print("parameters:", pformat(parameters))
    return {}
```

Now, you can run the experiment:

```
$ experitur run example.py
...
```

As you can see, `run` was called four times with every combination of [1,2] x [a,b].

### Multiple experiments

The Python file can contain multiple experiments:

```python
from experitur import experiment

@experiment(...)
def example1(trial):
    ...
    
@experiment(...)
def example2(trial):
    ...
```

### Experiment inheritance

One experiment may inherit the settings of another, using the `parent` parameter.

```python
from experitur import experiment

@experiment(...)
def example1(trial):
    ...
    
# Derived  with own entry point:
@experiment(parent=example1)
def example2(trial):
    ...
    
# Derived  with inherited entry point:
example3 = experiment("example3", parent=example2)
```

### Parameter substitution

`experitur` includes a recursive parameter substitution engine. Each value string is treated as a *recursive format string* and is resolved using the whole parameter set of a trial.

```python
@experiment(
    parameter_grid={
        "a1": [1],
        "a2": [2],
        "b": [1, 2],
        "a": ["{a_{b}}"],
    })
def example(trial):
    ...
```

```
$ experitur run parsub
...
```

This way, you can easily run complicated setups with settings that depend on other settings.

Recursive format strings work like `string.Formatter` with two exceptions:

1. **Recursive field names:** The field name itself may be a format string:

   ```
   format("{foo_{bar}}", bar="baz", foo_baz="foo") -> "foo"
   ```

2. **Literal output:** If the format string consist solely of a replacement field and does not contain a format specification, no to-string conversion is performed:

   ```
   format("{}", 1) -> 1
   ```

   This allows the use of format strings for non-string values.

#### Application

This feature is especially useful if you want to run your experiments for different datasets but need slightly different settings for each dataset.

Let's assume we have two datasets, "bees" and "flowers".

```python
@experiment(
    parameter_grid={
            "dataset": ["bees", "flowers"],
            "dataset_fn": ["/data/{dataset}/index.csv"],
            "bees-crop": [10],
            "flowers-crop": [0],
            "crop": ["{{dataset}-crop}"]
        }
)
def example(trial):
    ...
```

The experiment will be executed once for each dataset, with `trial["crop"]==10` for the "bees" dataset and `trial["crop"]==0` for the "flowers" dataset.

## The `trial` object

Every experiment receives a `trial` object that allows access to the parameters and meta-data of the trial.

Parameters are accessed with the `[]` operator (e.g. `trial["a"]`), meta-data is accessed with the `.` operator (e.g. `trial.wdir`).

### Access of parent data

...

## Files

When `experitur` executes a script, it creates the following file structure in the directory where the DOX file is located:

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
callable: example.experiment1
experiment: experiment1
id: experiment1/a-1_b-3
parameters: {a: 1, b: 3}
parent_experiment: null
result: null
success: true
time_end: 2019-06-07 14:22:41.697925
time_start: 2019-06-07 14:22:41.697837
wdir: examples/example/experiment1/a-1_b-3
```

Most items should be self-explanatory. `parameters` are the parameters passed to the entry point. `id` is derived from the parameters that are varied in the parameter grid. This way, you can easily interpret the file structure.

## Collecting results

Use `experitur collect script.py` to collect all the results (including parameters and metadata) of all trials of a lab book into a single CSV file located at `script/results.csv`.

## Calling functions and default parameters
Your experiment function might call other functions that have default parameters.
`experitur` gives you some utility functions that extract these default parameters adds them to the list of parameters.

For the following examples, let's assume `trial["p_a"]=1` and `trial["p_b"]=2`.

- `trial.without_prefix(prefix: str, parameters: dict) -> dict`: Extract all parameters that start with `prefix`.

  ```python
  trial.without_prefix("p_") == {"a": 1, "b": 2}
  ```

- `trial.apply(prefix: str, callable_: callable, *args, **kwargs)`: Call `callable_` with the parameters starting with `prefix`.

  ```python
  trial.apply("p_", fun, 10, c=20)
  # is the same as
  fun(10, a=1, b=2, c=20)
  ```

- `trial.record_defaults(prefix, [callable_,] **defaults)`: Set default values for parameters that were not set previously. Values in `defaults` override default parameters of `callable_`.

  ```python
  def foo(a=1, b=2, c=3):
      pass
  set_default_parameters("foo_", parameters, foo, c=4)
  # is the same as
  parameters.setdefault("foo_a", 1)
  parameters.setdefault("foo_b", 2)
  parameters.setdefault("foo_c", 4)
  ```

It is a good idea to make use of `set_default_parameters` and `apply_parameters` excessively. This way, your result files always contain the full set of parameters.

For a simple example, see [examples/example.py](examples/example.py).

## Installation

`experitur` is packaged on [PyPI](https://pypi.org/).

```shell
pip install experitur
```

Be warned that this package is currently under heavy development and anything might change any time!

## Examples

-  [examples/example.py](examples/example.py): A very basic example showing the workings of `set_default_parameters` and `apply_parameters`.
-  [examples/classifier.py](examples/classifier.py): Try different parameters of `sklearn.svm.SVC` to classify handwritten digits (the [MNIST](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) test set). Run the example, add more parameter values and see how `experitur` skips already existing configurations during the next run.

## Compatibility

`experitur` is [tested](https://travis-ci.org/moi90/experitur) with Python 3.5, 3.6 and 3.7.

## Similar software

- [Sacred](https://github.com/IDSIA/sacred)