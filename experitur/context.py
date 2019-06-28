import collections
import os.path
from contextlib import contextmanager

from experitur import trial
from experitur.errors import ExperiturError
from experitur.experiment import Experiment, StopExecution


class ContextError(ExperiturError):
    pass


class DependencyError(ContextError):
    pass


def _format_dependencies(experiments):
    msg = []
    for exp in experiments:
        msg.append("{} -> {}".format(exp, exp.parent))
    return "\n".join(msg)


def _order_experiments(experiments):
    experiments = set(experiments)

    # Include parents
    stack = collections.deque(experiments)

    while stack:
        exp = stack.pop()

        parent = exp.parent
        if parent is not None:
            if parent not in experiments:
                experiments.add(parent)
                stack.append(parent)

    done = set()
    experiments_ordered = []

    while experiments:
        # Get all without dependencies
        ready = {
            exp for exp in experiments if exp.parent is None or exp.parent in done}

        if not ready:
            raise DependencyError("Dependencies can not be satisfied:\n" +
                                  _format_dependencies(experiments))

        for exp in ready:
            experiments_ordered.append(exp)
            done.add(exp)
            experiments.remove(exp)

    return experiments_ordered


class Context:
    # Default configuration values
    _default_config = {
        "shuffle_trials": True,
        "skip_existing": True,
        "catch_exceptions": False,
    }

    def __init__(self, wdir=None, config=None):
        self.registered_experiments = []

        if wdir is None:
            self.wdir = "."
        else:
            self.wdir = wdir

        self.store = trial.FileTrialStore(self)

        # Configuration
        if config is None:
            self.config = self._default_config.copy()
        else:
            self.config = dict(self._default_config, **config)

    def _register_experiment(self, experiment):
        self.registered_experiments.append(experiment)

    def experiment(self, name=None, *, parameter_grid=None, parent=None, meta=None, active=True):
        """
        Experiment constructor.

        Can also be used as a decorator.
        """
        return Experiment(self, name=name, parameter_grid=parameter_grid, parent=parent, meta=meta, active=active)

    def run(self, experiments=None):
        """
        Run the specified experiments or all.
        """

        if experiments is None:
            experiments = self.registered_experiments

        # Now run the experiments in order
        ordered_experiments = _order_experiments(experiments)

        print("Running experiments:", ', '.join(
            exp.name for exp in ordered_experiments))
        for exp in ordered_experiments:
            try:
                exp.run()
            except Exception:
                if not self.config["catch_exceptions"]:
                    raise

    def update(self, experiments=None):
        """
        Update the specified experiments or all.
        """

        if experiments is None:
            experiments = self.registered_experiments

        # Update the experiments in order
        ordered_experiments = _order_experiments(experiments)

        print("Updating experiments:", ', '.join(
            exp.name for exp in ordered_experiments))
        for exp in ordered_experiments:
            try:
                exp.update()
            except Exception:
                if not self.config["catch_exceptions"]:
                    raise

    def collect(self, results_fn, failed=False):
        """
        Collect the results of all trials in 
        """
        data = {}
        for trial_id, trial in self.store.items():
            if not failed and not trial.data.get("success", False):
                # Skip failed trials if failed=False
                continue

            data[trial_id] = _prepare_trial_data(trial.data)

        import pandas as pd
        data = pd.DataFrame.from_dict(data, orient="index")
        data.index.name = "id"

        # TODO: Remove columns that are not serializable in CSV

        data.to_csv(results_fn)


def _prepare_trial_data(trial_data):
    result = {}

    for k, v in trial_data.items():
        if k in ("id", "parameters", "result"):
            continue
        result["meta_{}".format(k)] = v

    for k, v in trial_data.get("parameters", {}).items():
        result["{}".format(k)] = v

    trial_result = trial_data.get("result", {})

    if isinstance(trial_result, dict):
        for k, v in trial_result.items():
            result["{}_".format(k)] = v
    else:
        result["result_"] = trial_result

    return result


# Expose default context methods
default_context = Context()


def experiment(*args, **kwargs):
    return default_context.experiment(*args, **kwargs)


def run(*args, **kwargs):
    return default_context.run(*args, **kwargs)


@contextmanager
def push_context(ctx=None):
    """
    Context manager for creating a local context.

    Not thread-save.
    """
    global default_context
    old_ctx = default_context
    try:
        if ctx is None:
            ctx = Context()
        default_context = ctx
        yield default_context
    finally:
        default_context = old_ctx
