Current
=======

- Make scikit-learn and pandas optional dependencies.
- Change :code:`Experiment.__init__(parameter_grid)` to more flexible :code:`Experiment.__init__(parameters)`.
- Context: Remove experiment constructor. Remove :code:`push_context`.
- Remove :code:`Experiment.set_update`. The more general :code:`Experiment.command` should be used.