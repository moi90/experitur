Introduction
============

In this introduction, we will write a simple experiment to
find out which SVM kernel works best on MNIST data.

Defining the experiment
-----------------------

classifier.py:

.. literalinclude:: ../examples/classifier.py
    :language: python

Running the experiment
----------------------

.. image:: ../examples/classifier.gif
    :align: center

Looking at the results
----------------------

The results are saved in a folder with the same name as the description of experiment (DOX), in this case :code:`classifier/`.
For every trial, a subfolder :code:`classifier/<trial_id>` is created with a :code:`trial.yaml` file
containing the following data:

===========================   ========================================================================================================
:code:`experiment`            Experiment data.
:code:`id`                    Trial ID (:code:`<experiment name>/<trial id>`)
:code:`parameters`            Parameters as defined by the :py:class:`~experitur.parameters.ParameterGenerator`.
:code:`resolved_parameters`   Parameters after parameter substitution and filled in with values derived from
                              :py:meth:`~experitur.Trial.call` and :py:meth:`~experitur.Trial.record_defaults`.
:code:`result`                Result returned by the experiment function.
:code:`success`               TODO
:code:`time_start`            Time when the trial was started.
:code:`time_end`              Time when the trial ended.
:code:`wdir`                  Working directory of the trial.
===========================   ========================================================================================================

Running :code:`experitur collect classifier.py` will produce the following CSV file
that can be used to examine the results.

.. csv-table:: Results
   :file: ../examples/classifier.csv
   :header-rows: 1

As you can see, :code:`resolved_parameters` also contains the default values of :py:class:`sklearn.svm.SVC`.


Concepts
--------

The example uses the following concepts:

.. autosummary::
    :nosignatures:

    ~experitur.parameters.Grid
    ~experitur.Experiment
    ~experitur.Trial

The following methods of :py:class:`~experitur.Trial` were used:

.. autosummary::
    ~experitur.Trial.prefixed
    ~experitur.Trial.call


Further reading
---------------

- :doc:`installation`
- :doc:`parameter_substitution`
- :ref:`parameter-generators`
- :ref:`optimization`