.. experitur documentation master file, created by
   sphinx-quickstart on Sun Jun 23 10:14:45 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

experitur
=========

**experitur** automates machine learning and other computer science experiments
and stores the results in an easily accessible format.
It includes grid search, random search, parameter substitution, inheritance
and resuming aborted experiments.

.. image:: ../examples/simple.gif
    :align: center

The above example shows the execution of the following experiment:

.. literalinclude:: ../examples/simple.py
    :language: python

Read more about :py:class:`~experitur.Experiment`, :py:class:`~experitur.parameters.Grid` or the :doc:`command line reference<cli>`.


Contents
========

.. toctree::
    :maxdepth: 2

    introduction
    installation
    parameter_substitution
    cli
    api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`