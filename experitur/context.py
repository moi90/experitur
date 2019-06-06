import collections
from contextlib import contextmanager

from experitur import backends
from experitur.experiment import Experiment


def _format_dependencies(experiments):
    msg = []
    for exp in experiments:
        msg.append("{} -> {}".format(exp, exp.parent))
    return "\n".join(msg)


def _order_experiments(experiments):
    experiments = experiments.copy()
    done = set()
    experiments_ordered = []

    while experiments:
        # Get all without dependencies
        ready = {
            exp for exp in experiments if exp.parent is None or exp.parent in done}

        if not ready:
            raise ValueError("Dependencies can not be satisfied:\n" +
                             _format_dependencies(experiments))

        for exp in ready:
            experiments_ordered.append(exp)
            done.add(exp)
            experiments.remove(exp)

    return experiments_ordered


class Context:
    def __init__(self, wdir=None, backend=None, shuffle_trials=True, skip_existing=True):
        self.registered_experiments = []

        if wdir is None:
            self.wdir = "."
        else:
            self.wdir = wdir

        # if backend is None:
        #     self.backend = backends.FileBackend(self.wdir)
        # else:
        #     self.backend = backend

        self.shuffle_trials = shuffle_trials
        self.skip_existing = skip_existing

    def _register_experiment(self, experiment):
        self.registered_experiments.append(experiment)

    def experiment(self, name=None, *, parameter_grid=None, parent=None):
        """
        Experiment constructor.

        Can also be used as a decorator.
        """
        return Experiment(self, name=name, parameter_grid=parameter_grid, parent=parent)

    def run(self, experiments=None):
        """
        Run the specified experiments or all.
        """

        print("Context.run")

        if experiments is None:
            experiments = self.registered_experiments

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

        # Now run the experiments in order
        ordered_experiments = _order_experiments(experiments)

        print("Running experiments:", ', '.join(
            exp.name for exp in ordered_experiments))
        for exp in ordered_experiments:
            exp.run()


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
