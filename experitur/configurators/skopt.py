import logging
from collections import OrderedDict
from typing import Any, Container, Iterator, Mapping, Optional, Set

import skopt
import skopt.space
import skopt.utils
from unavailable_object import check_available

from experitur.core.configurators import ConfigurationSampler, Configurator
from experitur.core.context import get_current_context
from experitur.core.trial import Trial
from experitur.helpers.merge_dicts import merge_dicts
from experitur.util import format_parameters, unset


def convert_objective(value, maximize):
    if isinstance(value, tuple):
        # value is a (value, time) tuple
        value = value[0]

    if maximize:
        return -value
    return value


def convert_trial(
    trial: Trial,
    space: Mapping,
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

    point = skopt.utils.point_aslist(
        space,
        {k: v for k, v in parameters.items() if k in space},
    )

    return (point, result)


def _filter_results(optimizer: "skopt.Optimizer", results):
    return [r for r in results if r is not None and r[0] in optimizer.space]


def point_as_native_dict(space, point_as_list):
    params_dict = OrderedDict()
    for k, v in zip(sorted(space.keys()), point_as_list):
        if isinstance(space[k], skopt.space.Integer):
            v = int(v)
        elif isinstance(space[k], skopt.space.Real):
            v = float(v)
        params_dict[k] = v
    return params_dict


class _DimensionWrapper(Container):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def __contains__(self, x: object) -> bool:
        if x is unset:
            return False

        return x in self.dim

    def __str__(self) -> str:
        return str(self.dim)


class SKOpt(Configurator):
    """
    Configurator based on :py:class:`skopt.Optimizer`.

    Parameters:
        space (Mapping): Dictionary with parameters names as keys and distributions or lists of values.
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
            from experitur.configurators import SKOpt
            from scipy.stats.distributions import log

            @SKOpt({"a": [1,2], "b": expon()}, "y", 4)
            @Experiment()
            def example(parameters: Trial):
                print(parameters["a"], parameters["b"])

                return {"y": parameters["a"] * parameters["b"]}

        In this example, SKOpt will try to minimize :code:`y = a * b` using four evaluations.
    """

    Real = skopt.space.Real
    Integer = skopt.space.Integer
    Categorical = skopt.space.Categorical

    __str_attrs__ = ("space", "objective", "n_iter")

    def __init__(
        self,
        space: Mapping[str, Any],
        objective: str,
        n_iter: int,
        n_initial_points: int = 10,
        **kwargs,
    ):
        # Make sure that skopt is available
        check_available(skopt)

        self.space = {k: skopt.space.check_dimension(v) for k, v in space.items()}
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.objective = objective
        self.kwargs = kwargs

        self.logger = logging.getLogger(__name__)

    @property
    def parameter_values(self) -> Mapping[str, Container]:
        return {
            k: v.categories if isinstance(v, self.Categorical) else _DimensionWrapper(v)
            for k, v in self.space.items()
        }

    class _Sampler(ConfigurationSampler):
        configurator: "SKOpt"

        def sample(self, exclude: Optional[Set] = None) -> Iterator[Mapping]:
            space, exclude = self.prepare_values_exclude(
                self.configurator.space, exclude
            )

            ctx = get_current_context()
            experiment = ctx.current_experiment

            objective = self.configurator.objective

            if objective in experiment.maximize:
                maximize = True
            elif objective in experiment.minimize:
                maximize = False
            else:
                raise ValueError(
                    f"Could not determine if {objective} should be minimized or maximized. Specify minimize or maximize in Experiment."
                )

            for parent_configuration in self.parent.sample(exclude=exclude):
                # Retrieve all trials that match parent_configuration
                existing_trials = ctx.get_trials(
                    func=experiment.func,
                    parameters=parent_configuration.get("parameters", {}),
                ).filter(lambda trial: trial.result is not None)

                # Yield existing configurations
                existing_parameter_configurations = set(
                    tuple(
                        sorted(
                            (k, v) for k, v in trial.parameters.items() if k in space
                        )
                    )
                    for trial in existing_trials
                )

                for parameters in existing_parameter_configurations:
                    yield merge_dicts(parent_configuration, parameters=dict(parameters))

                while True:
                    optimizer = skopt.Optimizer(
                        skopt.utils.dimensions_aslist(self.configurator.space),
                        n_initial_points=self.configurator.n_initial_points,
                        **self.configurator.kwargs,
                    )

                    # Train model with existing trials
                    existing_trials = ctx.get_trials(
                        func=experiment.func,
                        parameters=parent_configuration.get("parameters", {}),
                    ).filter(lambda trial: trial.result is not None)

                    if len(existing_trials) >= self.configurator.n_iter:
                        break

                    results = _filter_results(
                        optimizer,
                        (
                            convert_trial(
                                trial,
                                self.configurator.space,
                                self.configurator.objective,
                                "ps" in optimizer.acq_func,
                                maximize,
                            )
                            for trial in existing_trials
                        ),
                    )

                    self.configurator.logger.info(
                        f"Training on {len(results):d} previous trials."
                    )

                    if results:
                        X, Y = zip(*results)
                        print(f"Current optimum: {convert_objective(min(Y), maximize)}")

                        optimizer.tell(X, Y)

                    if optimizer._n_initial_points > 0:
                        self.configurator.logger.info(
                            f"Random sampling. {optimizer._n_initial_points:d} random trials until optimizer fit."
                        )

                    # Get suggestion
                    # TODO: Save expected result (constant liar strategy) to allow parallel optimization
                    parameters = point_as_native_dict(
                        self.configurator.space, optimizer.ask()
                    )

                    self.configurator.logger.info(
                        f"Suggestion: {format_parameters(parameters)}"
                    )

                    yield merge_dicts(parent_configuration, parameters=parameters)
