# experitur
Automates machine learning and other computer experiments. Includes grid search and resuming aborted experiments.

## Lab notebook

Every experiment is described in a *lab notebook*. This is a text file with a YAML header, e.g. a Markdown file or a YAML file without further content:

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

Each experiment has a `run` setting. It points to a python function that receives a working directory and the parameters.

```yaml
---
# examle.md
id: example
run: "experitur.examples.echo:run"
parameter_grid:
    a: [1,2]
    b: [a,b]
---
```

```python
# echo.py
from pprint import pprint

def run(working_directory, parameters):
    print(working_directory)
    pprint(parameters)
```

Now, you can run the experiment:

```
$ experitur run example.md
Running example.md...
Independent parameters: ['a', 'b']
Trial 0: a-1_b-a
  0% (0/4) [               ] eta --:-- /
    a: 1
    b: a
example/example/a-1_b-a
{'a': 1, 'b': 'a'}
Trial 1: a-1_b-b
 25% (1/4) [###            ] eta --:-- -
    a: 1
    b: b
example/example/a-1_b-b
{'a': 1, 'b': 'b'}
Trial 2: a-2_b-a
 50% (2/4) [#######        ] eta 00:01 \
    a: 2
    b: a
example/example/a-2_b-a
{'a': 2, 'b': 'a'}
Trial 3: a-2_b-b
 75% (3/4) [###########    ] eta 00:01 |
    a: 2
    b: b
example/example/a-2_b-b
{'a': 2, 'b': 'b'}
Overall: 0.003s
  a-1_b-a: 0.000s (13%)
  a-2_b-a: 0.000s (8%)
  a-2_b-b: 0.000s (8%)
  a-1_b-b: 0.000s (8%)
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

`experitur` has a recursive parameter substitution engine. Each value string is treated as a *recursive format string* and is resolved using the whole parameter set of a trial.

```yaml
---
id: parsub
run: "experitur.examples.echo:run"
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

## Installation

Until `experitur` is packaged on [PyPI](https://pypi.org/), you can install it like so:

```shell
pip install git+https://github.com/moi90/experitur.git
```

Be warned that this package is currently under heavy development and anything might change any time!

## Compatibility
`experitur` is tested with Python 3.5, 3.6 and 3.7.

## Similar software

- [Sacred](https://github.com/IDSIA/sacred)