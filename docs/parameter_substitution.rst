Parameter substitution
======================

**experitur** includes a recursive parameter substitution engine.
Each value string is treated as a *recursive format string* and is resolved using the whole parameter set of a trial.

.. code-block:: python

    from experitur import Experiment

    @Experiment(
        parameters={
            "a_1": [1],
            "a_2": [2],
            "b": [1, 2],
            "a": ["{a_{b}}"],
        })
    def example(trial):
        ...

This way, you can easily run complicated setups with settings that depend on other settings.

Recursive format strings work like :py:class:`string.Formatter` with two exceptions:

1. **Recursive field names:** The field name itself may be a format string:

   .. code-block:: python
        
        format("{foo_{bar}}", bar="baz", foo_baz="foo") -> "foo"

2. **Literal output:** If the format string consist solely of a replacement field and does not contain a format specification, no to-string conversion is performed:

   .. code-block:: python
   
        format("{}", 1) -> 1

   This allows the use of format strings for non-string values.

Application
-----------

This feature is especially useful if you want to run your experiments for different datasets but need slightly different settings for each dataset.

Let's assume we have two datasets, "bees" and "flowers".

.. code-block:: python

    @Experiment(
        parameters={
                "dataset": ["bees", "flowers"],
                "dataset_fn": ["/data/{dataset}/index.csv"],
                "bees-crop": [10],
                "flowers-crop": [0],
                "crop": ["{{dataset}-crop}"]
            }
    )
    def example(parameters):
        ...

The experiment will be executed once for each dataset, with :code:`parameters["crop"]==10` for the "bees" dataset
and :code:`parameters["crop"]==0` for the "flowers" dataset.