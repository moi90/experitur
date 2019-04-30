import collections

from experitur.decorators import Experiment


def _format_dependencies(experiments):
    msg = []
    for exp in experiments:
        msg.append("{} -> {}".format(exp, exp.parent))
    return "\n".join(msg)


class Context:
    def __init__(self, shuffle_trials=True, skip_existing=True):
        self.registered_experiments = []
        self.shuffle_trials = shuffle_trials

    def register_experiment(self, experiment):
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
        done = set()

        while experiments:
            # Get all without dependencies
            ready = {
                exp for exp in experiments if exp.parent is None or exp.parent in done}

            if not ready:
                raise ValueError("Dependencies can not be satisfied:\n" +
                                 _format_dependencies(experiments))

            for exp in ready:
                exp.run()
                done.add(exp)
                experiments.remove(exp)


# Expose default context methods
default_context = Context()
experiment = default_context.experiment
run = default_context.run
