from typing import Mapping, Union, Any, Dict, List

from experitur.core.parameters import (
    ParameterGenerator,
    ParameterGeneratorIter,
    Grid,
    Multi,
)
from experitur.helpers.merge_dicts import merge_dicts

try:
    import skopt
    from skopt.utils import dimensions_aslist, point_asdict, point_aslist

    _skopt_available = True
except ImportError:  # pragma: no cover
    _skopt_available = False

try:
    from sklearn.model_selection import ParameterSampler

    _sklearn_available = True
except ImportError:  # pragma: no cover
    _sklearn_available = False


class _RandomSamplerIter(ParameterGeneratorIter):
    def __iter__(self):
        # Parameter names that are relevant in this context
        parameter_names = set(
            k
            for k in self.parameter_generator.param_distributions.keys()
            if k not in self.ignored_parameter_names
        )

        for parent_configuration in self.parent:
            # Firstly, produce existing configurations to give downstream samplers the chance to
            # produce missing sub-configurations.

            # Retrieve all trials that match parent_configuration
            existing_trials = self.experiment.ctx.store.match(
                callable=self.experiment.callable,
                parameters=parent_configuration.get("parameters", {}),
            )

            existing_params_set = set()
            for trial in existing_trials.values():
                existing_params_set.add(
                    tuple(
                        sorted(
                            (k, v)
                            for k, v in trial.data["parameters"].items()
                            if k in parameter_names
                        )
                    )
                )

            # Yield existing configurations
            for existing_params in existing_params_set:
                yield merge_dicts(
                    parent_configuration, parameters=dict(existing_params)
                )

            # Calculate n_iter as n_iter - already existing iterations
            n_iter = self.parameter_generator.n_iter - len(existing_params_set)

            for params in ParameterSampler(
                {
                    k: v
                    for k, v in self.parameter_generator.param_distributions.items()
                    if k not in self.ignored_parameter_names
                },
                n_iter,
            ):

                yield merge_dicts(parent_configuration, parameters=params)


class Random(ParameterGenerator):
    """
    Parameter sampler based on :py:class:`sklearn.model_selection.ParameterSampler`.

    If all parameters are presented as a list, sampling without replacement is performed.
    If at least one parameter is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous parameters.

    Parameters:
        param_distributions (dict): Dictionary with parameters names as keys and distributions or lists of values to try.
            Distributions must provide a rvs method for sampling (such as those from :py:mod:`scipy.stats`).
            If a list is given, it is sampled uniformly.
        n_iter (int): Number of parameter settings that are produced.

    Example:
        .. code-block:: python

            from experitur import Experiment
            from experitur.parameters import Random
            from scipy.stats.distributions import expon

            @Random({"a": [1,2], "b": expon()}, 4)
            @Experiment()
            def example(parameters: TrialParameters):
                print(parameters["a"], parameters["b"])

        This example will produce four runs, e.g. "1 0.898", "1 0.923", "2 1.878", and "2 1.038".

    """

    _iterator = _RandomSamplerIter
    _str_attr = ["n_iter"]

    def __init__(self, param_distributions: Dict[str, Union[List, Any]], n_iter: int):
        if not _sklearn_available:
            raise RuntimeError("scikit-learn is not available.")  # pragma: no cover

        self.param_distributions = param_distributions
        self.n_iter = n_iter

    @property
    def varying_parameters(self):
        return {
            k: v
            for k, v in self.param_distributions.items()
            if not isinstance(v, list) or len(v) > 1
        }

    @property
    def invariant_parameters(self):
        return {
            k: v
            for k, v in self.param_distributions.items()
            if isinstance(v, list) and len(v) <= 1
        }


class _SKOptIter(ParameterGeneratorIter):
    def __iter__(self):
        if not _skopt_available:
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
                callable=self.experiment.callable,
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
                for trial in existing_trials.values()
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
                    callable=self.experiment.callable,
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
                ]

                if results:
                    X, Y = zip(*results)
                    print(f"Current minimum: {min(Y)}")

                    optimizer.tell(X, Y)

                # Get suggestion
                # TODO: Save expected result (constant liar strategy) to allow parallel optimization
                parameters = point_asdict(
                    self.parameter_generator.search_space, optimizer.ask()
                )

                yield merge_dicts(parent_configuration, parameters=parameters)


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

            from experitur import Experiment
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
        self.search_space = search_space
        self.n_iter = n_iter
        self.objective = objective
        self.kwargs = kwargs

    @property
    def varying_parameters(self) -> Mapping:
        """Parameters in this sampler. Does not include parameters that do not vary."""
        return self.search_space
