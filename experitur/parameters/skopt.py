import logging
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Union

from experitur.core.parameters import ParameterGenerator, ParameterGeneratorIter
from experitur.helpers.merge_dicts import merge_dicts

try:
    import skopt
    from skopt.utils import dimensions_aslist, point_asdict, point_aslist
    import skopt.space

    _SKOPT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKOPT_AVAILABLE = False


class _SKOptIter(ParameterGeneratorIter):
    def __iter__(self):
        if not _SKOPT_AVAILABLE:
            raise RuntimeError("scikit-optimize is not available.")  # pragma: no cover

        # Parameter names that are relevant in this context
        parameter_names = set(
            k
            for k in self.parameter_generator.search_space.keys()
            if k not in self.ignored_parameter_names
        )

        for parent_configuration in self.parent:
            # Retrieve all trials that match parent_configuration
            existing_trials = self.experiment.ctx.store.match(
                func=self.experiment.func,
                parameters=parent_configuration.get("parameters", {}),
            )

            # Yield existing configurations
            existing_parameter_configurations = set(
                tuple(
                    sorted(
                        (k, v)
                        for k, v in trial.data["parameters"].items()
                        if k in parameter_names
                    )
                )
                for trial in existing_trials
            )

            for parameters in existing_parameter_configurations:
                yield merge_dicts(parent_configuration, parameters=dict(parameters))

            # Calculate n_iter as n_iter - already existing iterations
            n_iter = self.parameter_generator.n_iter - len(
                existing_parameter_configurations
            )

            for _ in range(n_iter):
                optimizer = skopt.Optimizer(
                    dimensions_aslist(self.parameter_generator.search_space),
                    **self.parameter_generator.kwargs,
                )

                # Train model
                existing_trials = self.experiment.ctx.store.match(
                    func=self.experiment.func,
                    parameters=parent_configuration.get("parameters", {}),
                )

                # TODO: Record timing for time based `acq_func`s
                results = [
                    (
                        point_aslist(
                            self.parameter_generator.search_space,
                            {
                                k: v
                                for k, v in trial.data["parameters"].items()
                                if k in parameter_names
                            },
                        ),
                        trial.data["result"][self.parameter_generator.objective],
                    )
                    for trial in existing_trials.values()
                    if trial.data.get("result", None)
                ]

                self.parameter_generator.logger.info(
                    f"Training on {len(results):d} previous trials."
                )

                if results:
                    X, Y = zip(*results)
                    print(f"Current minimum: {min(Y)}")

                    optimizer.tell(X, Y)

                if optimizer._n_initial_points > 0:
                    self.parameter_generator.logger.info(
                        f"Random sampling. {optimizer._n_initial_points:d} random trials until optimizer fit."
                    )

                # Get suggestion
                # TODO: Save expected result (constant liar strategy) to allow parallel optimization
                parameters = point_as_native_dict(
                    self.parameter_generator.search_space, optimizer.ask()
                )

                self.parameter_generator.logger.info(f"Suggestion: {parameters}")

                yield merge_dicts(parent_configuration, parameters=parameters)


def point_as_native_dict(search_space, point_as_list):
    params_dict = OrderedDict()
    for k, v in zip(sorted(search_space.keys()), point_as_list):
        if isinstance(search_space[k], skopt.space.Integer):
            v = int(v)
        elif isinstance(search_space[k], skopt.space.Real):
            v = float(v)
        params_dict[k] = v
    return params_dict


class SKOpt(ParameterGenerator):
    """
    Parameter sampler based on :py:class:`skopt.Optimizer`.

    Parameters:
        search_space (Mapping): Dictionary with parameters names as keys and distributions or lists of values.
            Each parameter can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for :py:class:`~skopt.space.space.Real` or :py:class:`~skopt.space.space.Integer` dimensions),
            - a `(lower_bound, upper_bound, prior)` tuple (for :py:class:`~skopt.space.space.Real` dimensions),
            - as a list of categories (for :py:class:`~skopt.space.space.Categorical` dimensions), or
            - an instance of a :py:class:`~skopt.space.space.Dimension` object (:py:class:`~skopt.space.space.Real`, :py:class:`~skopt.space.space.Integer` or :py:class:`~skopt.space.space.Categorical`).
        objective (str): Name of the result that will be optimized.
        n_iter (int): Number of evaluations to find the optimum.
        **kwargs: Additional arguments for :py:class:`skopt.Optimizer`.


    Example:
        .. code-block:: python

            from experitur import Experiment, TrialParameters
            from experitur.parameters import SKOpt
            from scipy.stats.distributions import log

            @SKOpt({"a": [1,2], "b": expon()}, "y", 4)
            @Experiment()
            def example(parameters: TrialParameters):
                print(parameters["a"], parameters["b"])

                return {"y": parameters["a"] * parameters["b"]}

        In this example, SKOpt will try to minimize :code:`y = a * b` using four evaluations.
    """

    _iterator = _SKOptIter
    _str_attr = ["search_space", "objective", "n_iter"]

    def __init__(
        self, search_space: Mapping[str, Any], objective: str, n_iter: int, **kwargs
    ):
        self.search_space = search_space = {
            k: skopt.space.check_dimension(v) for k, v in search_space.items()
        }
        self.n_iter = n_iter
        self.objective = objective
        self.kwargs = kwargs

        self.logger = logging.getLogger(__name__)

    @property
    def varying_parameters(self) -> Mapping:
        """Parameters in this sampler. Does not include parameters that do not vary."""
        return self.search_space
