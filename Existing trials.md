## Existing trials

If the same lab notebook is run a second time, `experitur` skips existing trials by default.

### Running
P_to_run, P_existing
- Skip existing
  - ~~Same `trial_id`? Not sufficient if parameters are added or deleted during trials~~
  - ~~`parameters == parameter_pre`? Not sufficient alone, if default parameters change between runs or the value of a parameter deviates in a second run. Might skip too many.~~
  - `parameters subseteq parameter_post`? Not sufficient alone, if default parameters change between runs. -> We can't track changes of default parameters without actually running the trial.
    -> Able to detect deviation from default parameters. (`subseteq` =:= P_to_run.keys() is a nonempty subset of P_to_run.keys() and the values for each matching key are the same)
  - ~~`parameters == parameter_post`? Wrong, because default parameters are added. Might skip none at all.~~
- Trial ID already taken? Include existing trial in calculation of varying parameters.
- Redo existing. 

### Cleaning

- All
- Failed (`success is False` or trial directory is empty)
- [Experiment]

## Backend

- Retrieve trial data by id (`'<experiment>/<trial>'`)
- Generate trial id given parameter set (experiment, parameters)
- Retrieve trial data by parameter set

