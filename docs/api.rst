API reference
=============

Experiment
----------

.. autoclass:: experitur.Experiment
    :members:

    .. automethod:: experitur.Experiment.__call__


.. _parameter-generators:

Parameter generators
--------------------

Parameter generators can be supplied using the :code:`parameters` parameter of :py:class:`~experitur.Experiment`
or as :std:term:`decorators<decorator>`.
Simple parameter grids can be passed as a :py:class:`dict` to the :code:`parameters` parameter.

.. autoclass:: experitur.parameters.Grid

.. autoclass:: experitur.parameters.Random

Stacking parameter generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple parameter generators can be stacked.

.. code-block:: python

    from experitur import Experiment
    from experitur.parameters import Grid, Random
    from scipy.stats.distributions import expon

    @Grid({"a": [1,2,3]}
    @Random({"b": expon()}, 4)
    @Experiment()
    def example(parameters: TrialParameters):
        print(parameters["a"], parameters["b"])

For every value of :code:`a`, four values will be drawn for :code:`b`.

.. _optimization:

Optimization
~~~~~~~~~~~~

:py:class:`experitur.parameters.SKOpt` is a wrapper for :py:class:`skopt.Optimizer`.

.. note:: You have to install the :code:`skopt` dependency using pip:

    .. code-block:: sh

        pip install scikit-optimize

:code:`skopt` allows :py:class:`~skopt.space.space.Real`,
:py:class:`~skopt.space.space.Integer` and :py:class:`~skopt.space.space.Categorical` variables.

.. autoclass:: experitur.parameters.SKOpt

Combining :py:class:`~experitur.parameters.SKOpt` with other generators allows a sophisticated exploration of the parameter space:

.. literalinclude:: ../examples/optimization.py
    :language: python

For every combination of :code:`a` and :code:`b`, ten value combinations for :code:`x` and :code:`y` are produced in order to minimize :code:`z`,
but :code:`z` is averaged across three runs.

Trial Parameters
----------------

.. autoclass:: experitur.TrialParameters
    :members:

    .. automethod:: experitur.TrialParameters.__getitem__
    .. automethod:: experitur.TrialParameters.__getattr__


