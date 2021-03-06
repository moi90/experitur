import logging
from collections import OrderedDict
from typing import Any, Mapping, TYPE_CHECKING

from experitur.core.parameters import ParameterGenerator, ParameterGeneratorIter
from experitur.core.trial import Trial
from experitur.helpers.merge_dicts import merge_dicts
from unavailable_object import UnavailableObject, check_available

try:
    import skopt
    import skopt.utils as skopt_utils
    import skopt.space as skopt_space
except ImportError:
    if not TYPE_CHECKING:
        skopt = UnavailableObject("skopt")
        skopt_utils = UnavailableObject("skopt.utils")
        skopt_space = UnavailableObject("skopt.space", none_child=True)


def convert_objective(value, maximize):
    if maximize:
        return -value
    return value


def convert_trial(
    trial: Trial,
    search_space: Mapping,
    objective: str,
    include_duration=False,
    maximize=False,
):
    """Convert a trial to a tuple (parameters, (result, time)) or None."""

    result = getattr(trial, "result", {}).get(objective, None)
    parameters = getattr(trial, "resolved_parameters", None)
    time_start = getattr(trial, "time_start", None)
    time_end = getattr(trial, "time_end", None)

    if None in (result, parameters, time_start, time_end):
        return None

    result = convert_objective(result, maximize)

    if include_duration:
        result = (result, (time_end - time_start).total_seconds())

    point = skopt_utils.point_aslist(
        search_space,
        {k: v for k, v in parameters.items() if k in search_space},
    )

    return (point, result)


def _filter_results(optimizer: "skopt.Optimizer", results):
    return [r for r in results if r is not None and r[0] in optimizer.space]


class _SKOptIter(ParameterGeneratorIter):
    def __iter__(self):
        objective = self.parameter_generator.objective

        if objective in self.experiment.maximize:
            maximize = True
        elif objective in self.experiment.minimize:
            maximize = False
        else:
            raise ValueError(
                f"Could not determine if {objective} should be minimized or maximized. Specify minimize or maximize in Experiment."
            )

        # Parameter names that are relevant in this context
        parameter_names = set(
            k
            for k in self.parameter_generator.search_space.keys()
            if k not in self.ignored_parameter_names
        )

        for parent_configuration in self.parent:
            # Retrieve all trials that match parent_configuration
            existing_trials = self.experiment.ctx.get_trials(
                func=self.experiment.func,
                parameters=parent_configuration.get("parameters", {}),
            )

            # Yield existing configurations
            existing_parameter_configurations = set(
                tuple(
                    sorted(
                        (k, v)
                        for k, v in trial.parameters.items()
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
                    skopt_utils.dimensions_aslist(
                        self.parameter_generator.search_space
                    ),
                    n_initial_points=self.parameter_generator.n_initial_points,
                    **self.parameter_generator.kwargs,
                )

                # Train model
                existing_trials = self.experiment.ctx.get_trials(
                    func=self.experiment.func,
                    parameters=parent_configuration.get("parameters", {}),
                )

                results = _filter_results(
                    optimizer,
                    (
                        convert_trial(
                            trial,
                            self.parameter_generator.search_space,
                            self.parameter_generator.objective,
                            "ps" in optimizer.acq_func,
                            maximize,
                        )
                        for trial in existing_trials
                    ),
                )

                self.parameter_generator.logger.info(
                    f"Training on {len(results):d} previous trials."
                )

                if results:
                    X, Y = zip(*results)
                    print(f"Current optimum: {convert_objective(min(Y), maximize)}")

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
        if isinstance(search_space[k], skopt_space.Integer):
            v = int(v)
        elif isinstance(search_space[k], skopt_space.Real):
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
        objective (str): Name of the result that will be minimized. If starts with "-", the following name will be maximized instead.
        n_iter (int): Number of evaluations to find the optimum.
        n_initial_points (int): Number of points sampled at random before actual optimization.
        **kwargs: Additional arguments for :py:class:`skopt.Optimizer`.


    Example:
        .. code-block:: python

            from experitur import Experiment, Trial
            from experitur.parameters import SKOpt
            from scipy.stats.distributions import log

            @SKOpt({"a": [1,2], "b": expon()}, "y", 4)
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

                return {"y": parameters["a"] * parameters["b"]}

        In this example, SKOpt will try to minimize :code:`y = a * b` using four evaluations.
    """

    AVAILABLE = not isinstance(skopt, UnavailableObject)

    Real = skopt_space.Real
    Integer = skopt_space.Integer
    Categorical = skopt_space.Categorical

    _iterator = _SKOptIter
    _str_attr = ["search_space", "objective", "n_iter"]

    def __init__(
        self,
        search_space: Mapping[str, Any],
        objective: str,
        n_iter: int,
        n_initial_points: int = 10,
        **kwargs,
    ):
        # Make sure that skopt is available
        check_available(skopt)

        self.search_space = search_space = {
            k: skopt_space.check_dimension(v) for k, v in search_space.items()
        }
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.objective = objective
        self.kwargs = kwargs

        self.logger = logging.getLogger(__name__)

    @property
    def varying_parameters(self) -> Mapping:
        """Parameters in this sampler. Does not include parameters that do not vary."""
        return self.search_space
