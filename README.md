# experitur

[![](https://img.shields.io/pypi/v/experitur.svg?style=flat)](https://pypi.org/project/experitur/) [![Build Status](https://travis-ci.org/moi90/experitur.svg?branch=master)](https://travis-ci.org/moi90/experitur) [![codecov](https://codecov.io/gh/moi90/experitur/branch/master/graph/badge.svg)](https://codecov.io/gh/moi90/experitur) ![](https://img.shields.io/pypi/pyversions/experitur.svg?style=flat)

Automates machine learning and other computer experiments. Includes grid search and resuming aborted experiments. No lock-in, all your data is easily accessible in a text-based, machine-readable format.

## Lab notebook

Every experiment is described in a *lab book*. This is a text file with a YAML header, e.g. a Markdown file or a YAML file without further content:

```yaml
---
# In this part of the document called the "experiment section", enclosed by "---", you describe the experiment(s).
id: example
parameter_grid:
    parameter_1: [1,2,3]
    parameter_2: [a,b,c]
---
# An example experiment
In this part of the document, you can write down any content you like. Markdown files are allowed to contain a YAML header, so this could be Markdown.
```

### Parameter grid

The core of an experiment is its *parameter grid*. It works like [`sklearn.model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html). Each parameter has a list of values that it can take. A number of *trials* is generated from the cross product of the values of each parameter.

### Run function

Each experiment has a `run` setting (unless it is an *abstract experiment*). It is a string pointing to a python function (i.e. `<fully.qualified.name>:<function_name>`). Upon execution, the function gets called with the working directory and the parameters of the current trial. It may return a result dictionary.

**Signature:** `(working_directory: str, parameters: dict) -> dict`

```yaml
---
# examle_labbook.md
id: example_experiment
run: "echo:run"
parameter_grid:
    a: [1,2]
    b: [a,b]
---
```

```python
# echo.py
from pprint import pformat

def run(working_directory, parameters):
    print("working_directory:", working_directory)
    print("parameters:", pformat(parameters))
    return {}
```

Now, you can run the experiment:

```
$ experitur run example_labbook.md
Running example_labbook.md...
Independent parameters: ['a', 'b']
Trial a-1_b-a:   0%|                                                                                                                                           | 0/4 [00:00<?, ?/s]
    a: 1
    b: a
working_directory: example_labbook/example_experiment/a-1_b-a
parameters: {'a': 1, 'b': 'a'}
Trial a-1_b-b:  50%|█████████████████████████████████████████████████████████████████▌                                                                 | 2/4 [00:01<00:01,  1.98/s]
    a: 1
    b: b
working_directory: example_labbook/example_experiment/a-1_b-b
parameters: {'a': 1, 'b': 'b'}
Trial a-2_b-a:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████▎                                | 3/4 [00:02<00:00,  1.52/s]
    a: 2
    b: a
working_directory: example_labbook/example_experiment/a-2_b-a
parameters: {'a': 2, 'b': 'a'}
Trial a-2_b-b: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.31/s]
    a: 2
    b: b
working_directory: example_labbook/example_experiment/a-2_b-b
parameters: {'a': 2, 'b': 'b'}
Overall: 4.035s
  a-1_b-a: 1.009s (25%)
  a-1_b-b: 1.008s (24%)
  a-2_b-a: 1.007s (24%)
  a-2_b-b: 1.007s (24%)
```

As you can see, `run` was called four times with every combination of [1,2] x [a,b].

### Multiple experiments

The experiment section can hold multiple experiments in a list:

```yaml
---
- id: experiment_1
    parameter_grid:
        ...
- id: experiment_2
    parameter_grid:
        ...
---
```

### Experiment inheritance

One experiment may inherit the settings of another, using the `base` property:

```yaml
---
- id: experiment_1
    parameter_grid:
        a: [1, 2, 3]
- id: experiment_2
    base: experiment_1
    parameter_grid:
        b: [x, y, z]
        # In effect, experiment_2 also a parameter 'a' that takes the values 1,2,3.
---
```

### Parameter substitution

`experitur` includes a recursive parameter substitution engine. Each value string is treated as a *recursive format string* and is resolved using the whole parameter set of a trial.

```yaml
---
id: parsub
run: "echo:run"
parameter_grid:
    a_1: [foo]
    a_2: [bar]
    a: ["{a_{b}}"]
    b: [1,2]
---
```

```
$ experitur run parsub.md
Running parsub.md...
Independent parameters: ['b']
Trial 0: b-1
  0% (0/2) [               ] eta --:-- /
    a: foo
    a_1: foo
    a_2: bar
    b: 1
parsub/parsub/b-1
{'a': 'foo', 'a_1': 'foo', 'a_2': 'bar', 'b': 1}
Trial 1: b-2
 50% (1/2) [#######        ] eta --:-- -
    a: bar
    a_1: foo
    a_2: bar
    b: 2
parsub/parsub/b-2
{'a': 'bar', 'a_1': 'foo', 'a_2': 'bar', 'b': 2}
Overall: 0.002s
  b-1: 0.000s (18%)
  b-2: 0.000s (14%)
```

This way, you can easily run complicated setups with settings that depend on other settings.

Recursive format strings work like `string.Formatter` with two excpetions:

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
When `experitur` executes a lab book, it creates the following file structure in the directory where the lab book is located:

```
/
+- lab_book.md
+- lab_book/
|  +- experiment1_id/
|  |  +- trial1_id/
|  |  |  +- experitur.yaml
|  |  ...
|  ...
```

`<lab_book_name>/<experiment_id>/<trial_id>/experitur.yaml` contains the parameters and the results from a trial, e.g.:

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

Most items should be self-explanatory. `parameters_pre` are the parameters passed to the run function,  `parameters_post` are the parameters after the run function had the chance to update them. `trial_id` is derived from the parameters that are varied in the parameter grid. This way, you can easily interpret the file structure.

## Collecting results

Use `experitur collect <lab_book>.md` to collect all the results (including parameters and metadata) of all trials of a lab book into a single CSV file located at `<lab_book>/results.csv`.

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

## Compatibility
`experitur` is [tested](https://travis-ci.org/moi90/experitur) with Python 3.5, 3.6 and 3.7.

## Similar software

- [Sacred](https://github.com/IDSIA/sacred)