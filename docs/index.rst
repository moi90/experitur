.. experitur documentation master file, created by
   sphinx-quickstart on Sun Jun 23 10:14:45 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to experitur's documentation!
=====================================

**experitur** automates machine learning and other computer science experiments
and stores the results in an easily accessible format.
It includes grid search, parameter substitution
and resuming aborted experiments.

.. contents:: Contents
    :local:

Installation
------------

experitur is packaged on PyPI and can be installed with pip:

.. code-block:: sh

    pip install experitur

Be warned that this package is currently under heavy development
and anything might change any time!

Getting started
---------------

experitur is very easy to use. Just create a Python file where you describe
your experiments like so:

.. literalinclude:: ../examples/simple.py
    :language: python

You can then run your experiment:

.. code-block:: sh

    experitur run simple.py



.. toctree::
    :maxdepth: 2

    getting_started
    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
