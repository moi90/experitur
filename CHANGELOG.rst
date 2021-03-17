Current (will become 2.0.0)
===========================

- Merge Trial and TrialParameters.
- Add :code:`TrialCollection`.
- Add :code:`Trial.flush()`.
- Add reloading of experiment file after execution.
- Add "volatile" Experiments.
- Add :code:`Trial.log`.
- Fix endless recusion error in :py:func:`redirect_stdout`.
- Add :code:`Trial.choice`.
- Add :code:`Trial.prefixed`.
- Change :code:`Trial.apply` to :code:`Trial.call`.
- Remove :code:`Experiment.post_grid`.
- Make scikit-learn and pandas optional dependencies.
- Change :code:`Experiment.__init__(parameter_grid)` to more flexible :code:`Experiment.__init__(parameters)`.
- Context: Remove experiment constructor. Remove :code:`push_context`.
- Remove :code:`Experiment.set_update`. The more general :code:`Experiment.command` should be used.
- Move context, experiment, samplers, trial to core.
- Add :code:`ParameterGenerator` decorators: :code:`Grid`, :code:`Random`, :code:`SKOpt`.
- Change :code:`experiment` decorator to :code:`Experiment`.
- 