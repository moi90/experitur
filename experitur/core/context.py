import collections
import os.path
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional, Union

from experitur.core.trial import TrialCollection, Trial
from experitur.errors import ExperiturError

if TYPE_CHECKING:  # pragma: no cover
    from experitur.core.experiment import Experiment


class ContextError(ExperiturError):
    pass


class DependencyError(ContextError):
    pass


def _format_dependencies(experiments):
    msg = []
    for exp in experiments:
        msg.append("{} -> {}".format(exp, exp.parent))
    return "\n".join(msg)


def _order_experiments(experiments) -> List["Experiment"]:
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
        ready = {exp for exp in experiments if exp.parent is None or exp.parent in done}

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

        # Configuration
        if config is None:
            self.config = self._default_config.copy()
        else:
            self.config = dict(self._default_config, **config)

        self.writable = writable

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
        for exp in ordered_experiments:
            exp.run()

    def collect(self, results_fn: Union[str, Path], failed=False):
        """
        Collect the results of all trials in this context.

        Parameters:
            results_fn (str or Path): Path where the result should be written.
            failed (boolean): Include failed trials. (Default: False)
        """

        if isinstance(results_fn, Path):
            results_fn = str(results_fn)

        try:
            from pandas import json_normalize
        except ImportError:
            try:
                from pandas.io.json import json_normalize
            except ImportError:  # pragma: no cover
                raise RuntimeError("pandas is not available.")

        data = []
        for trial_data in self.store.values():
            if not failed and not trial_data.get("success", False):
                # Skip failed trials if failed=False
                continue

            data.append(trial_data)

        data = json_normalize(data, max_level=1).set_index("id")

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
        return TrialCollection(
            [
                Trial(td, self.store)
                for td in self.store.match(
                    func=func, resolved_parameters=parameters, experiment=experiment
                )
            ]
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


_context_stack: List[Context] = []


def get_current_context() -> Context:
    return _context_stack[-1]
