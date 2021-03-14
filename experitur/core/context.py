import collections
import contextlib
import os.path
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional, Union

from experitur.core.root_trial_collection import RootTrialCollection
from experitur.core.trial import Trial, TrialCollection
from experitur.errors import ExperiturError

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment


class ContextError(ExperiturError):
    pass


class DependencyError(ContextError):
    pass


def _format_dependencies(experiments: List["Experiment"]):
    msg = []
    for exp in experiments:
        msg.append("{} -> {}".format(exp, ", ".join(str(d) for d in exp.depends_on)))
    return "\n".join(msg)


def _order_experiments(experiments) -> List["Experiment"]:
    experiments = set(experiments)

    # Include parents
    stack = collections.deque(experiments)

    while stack:
        exp: "Experiment" = stack.pop()

        for dependency in exp.depends_on:
            if dependency not in experiments:
                experiments.add(dependency)
                stack.append(dependency)

    done = set()
    experiments_ordered = []

    while experiments:
        # Get all without dependencies
        ready = {
            exp
            for exp in experiments
            if not exp.depends_on or all(d in done for d in exp.depends_on)
        }

        if not ready:
            raise DependencyError(
                "Dependencies can not be satisfied:\n"
                + _format_dependencies(experiments)
            )

        for exp in ready:
            experiments_ordered.append(exp)
            done.add(exp)
            experiments.remove(exp)

    return experiments_ordered


class Context:
    """
    experitur context.

    Args:
        wdir (str): Working directory.
        config (dict, optional): Configuration dict.
        writable (bool, default False): Set this context writable.
    """

    # Default configuration values
    _default_config = {
        "skip_existing": True,
        "catch_exceptions": False,
    }

    def __init__(
        self, wdir: str = None, config: Optional[Mapping] = None, writable: bool = False
    ):
        self.registered_experiments = []

        if wdir is None:
            self.wdir = "."
        else:
            self.wdir = wdir

        # Import here to break dependency cycle
        from experitur.core.trial_store import FileTrialStore

        self.store = FileTrialStore(self)
        self.trials = RootTrialCollection(self)

        # Configuration
        if config is None:
            self.config = self._default_config.copy()
        else:
            self.config = dict(self._default_config, **config)

        self.writable = writable

        self._current_trial = None

    def _register_experiment(self, experiment):
        self.registered_experiments.append(experiment)

    def run(self, experiments=None):
        """
        Run the specified experiments or all.
        """

        if not self.writable:
            raise ContextError("No experiments can be run in a read-only context.")

        if experiments is None:
            experiments = self.registered_experiments

        # Now run the experiments in order
        ordered_experiments = _order_experiments(experiments)

        print(
            "Running experiments:", ", ".join(exp.name for exp in ordered_experiments)
        )

        try:
            for exp in ordered_experiments:
                exp.run()
        finally:
            # If no more trials are running (also in other processes) clear the stop signal
            running_trials = self.trials.filter(
                lambda trial: not trial.is_failed and not trial.is_successful
            )
            if not running_trials:
                # Clear stop
                self.stop(False)

    def collect(self, results_fn: Union[str, Path], failed=False):
        """
        Collect the results of all trials in this context.

        Parameters:
            results_fn (str or Path): Path where the result should be written.
            failed (boolean): Include failed trials. (Default: False)
        """

        if isinstance(results_fn, Path):
            results_fn = str(results_fn)

        data = self.trials.to_pandas()

        data.to_csv(results_fn)

    def get_experiment(self, name) -> "Experiment":
        """
        Get an experiment by its name.

        Args:
            name: Experiment name.

        Returns:
            A :class:`Experiment` instance.

        Raises:
            :obj:`KeyError` if no experiment with this name is found.
        """
        try:
            return [e for e in self.registered_experiments if e.name == name][0]
        except IndexError:
            print(self.registered_experiments)
            raise KeyError(name) from None

    def get_trials(
        self, func=None, parameters=None, experiment=None
    ) -> TrialCollection:
        return self.trials.match(
            func=func, resolved_parameters=parameters, experiment=experiment
        )

    def do(self, target, cmd, cmd_args):
        experiment_name = target.split("/")[0]

        experiment = self.get_experiment(experiment_name)

        return experiment.do(cmd, target, cmd_args)

    def __enter__(self):
        # Push self to context stack
        _context_stack.append(self)

        return self

    def __exit__(self, *_):
        # Pop self from context stack
        item = _context_stack.pop()

        assert item is self

    def get_trial_wdir(self, trial_id):
        return os.path.normpath(os.path.join(self.wdir, os.path.normpath(trial_id)))

    def get_trial(self, trial_id) -> Trial:
        return Trial(self.store[trial_id], self.store)

    @property
    def current_trial(self):
        return self._current_trial

    @contextlib.contextmanager
    def set_current_trial(self, trial: Trial):
        try:
            self._current_trial = trial
            yield
        finally:
            self._current_trial = None

    def stop(self, stop=True):
        """Save/clear stop signal."""

        flag_fn = os.path.join(self.wdir, "stop")

        if stop:
            with open(flag_fn, "w"):
                pass
        else:
            try:
                os.unlink(flag_fn)
            except Exception:  # pylint: disable=broad-except
                pass

    def should_stop(self):
        flag_fn = os.path.join(self.wdir, "stop")

        return os.path.isfile(flag_fn)


_context_stack: List[Context] = []


def get_current_context() -> Context:
    return _context_stack[-1]
