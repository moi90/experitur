# experitur

[![](https://img.shields.io/pypi/v/experitur.svg?style=flat)](https://pypi.org/project/experitur/) [![Build Status](https://travis-ci.org/moi90/experitur.svg?branch=master)](https://travis-ci.org/moi90/experitur) [![codecov](https://codecov.io/gh/moi90/experitur/branch/master/graph/badge.svg)](https://codecov.io/gh/moi90/experitur) ![](https://img.shields.io/pypi/pyversions/experitur.svg?style=flat)

Automates machine learning and other computer experiments. Includes grid search and resuming aborted experiments. No lock-in, all your data is easily accessible in a text-based, machine-readable format.

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
$ experitur run example
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

## Files
When `experitur` executes a script, it creates the following file structure in the directory where the lab book is located:

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
experiment_id: example_experiment
parameters_post: {a: 1, b: a}
parameters_pre: {a: 1, b: a}
result: {}
success: true
time_end: 2019-01-31 13:50:51.003637
time_start: 2019-01-31 13:50:50.002264
trial_id: a-1_b-a
```

Most items should be self-explanatory. `parameters` are the parameters passed to the run function,  `parameters_post` are the parameters after the run function had the chance to update them. `trial_id` is derived from the parameters that are varied in the parameter grid. This way, you can easily interpret the file structure.

## Collecting results

Use `experitur collect script` to collect all the results (including parameters and metadata) of all trials of a lab book into a single CSV file located at `script/results.csv`.

## Calling functions and default parameters
Your `run` function might call other functions that have default parameters.
`experitur` gives you some utility functions that extract these default parameters adds them to the list of parameters.

- `extract_parameters(prefix: str, parameters: dict) -> dict`: Extract all parameters that start with `prefix`.

  ```python
  extract_parameters("p_", {"p_a": 1, "p_b": 2}) == {"a": 1, "b": 2}
  ```

- `apply_parameters(prefix: str, parameters: dict, callable_: callable, *args, **kwargs)`: Call `callable_` with the parameters starting with `prefix`.

  ```python
  apply_parameters("p_", {"p_a": 1, "p_b": 2}, fun, 10, c=20)
  # is the same as
  fun(10, a=1, b=2, c=20)
  ```

- `set_default_parameters(prefix, parameters, [callable_,] **defaults)`: Set default values for parameters that were not set previously. Values in `defaults` override default parameters of `callable_`.

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

For a simple example, see [examples/str_split.md](examples/str_split.md).

## Installation

`experitur` is packaged on [PyPI](https://pypi.org/).

```shell
pip install experitur
```

Be warned that this package is currently under heavy development and anything might change any time!

## Examples

-  [examples/str_split.md](examples/str_split.md): A very basic example showing the workings of `set_default_parameters` and `apply_parameters`.
-  [examples/mnist.md](examples/mnist.md): Try different parameters of `sklearn.svm.SVC` to classify handwritten digits (the [MNIST](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) test set). Run the example, add more parameter values and see how `experitur` skips already existing configurations during the next run.

## Compatibility

`experitur` is [tested](https://travis-ci.org/moi90/experitur) with Python 3.5, 3.6 and 3.7.

## Similar software

- [Sacred](https://github.com/IDSIA/sacred)