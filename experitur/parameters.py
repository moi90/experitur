from typing import Mapping

from experitur.core.parameters import *
from experitur.helpers.merge_dicts import merge_dicts

try:
    import skopt
    from skopt.utils import dimensions_aslist, point_asdict, point_aslist

    _skopt_available = True
except ImportError:
    _skopt_available = False


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
    Parameter sampler based on py:class:`sklearn.model_selection.ParameterSampler`.
    """

    _iterator = _RandomSamplerIter
    _str_attr = ["n_iter"]

    def __init__(self, param_distributions: Dict, n_iter):
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
            raise RuntimeError("scikit-optimize is not available.")

        print("OptimizerIter.__init__")

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

            print("n_iter", n_iter)

            for _ in range(n_iter):
                optimizer = skopt.Optimizer(
                    dimensions_aslist(self.parameter_generator.search_space),
                    **self.parameter_generator.kwargs,
                )

                # Train model
                existing_trials = self.experiment.ctx.store.match(
                    callable=self.experiment.callable,
                    parameters=parent_configuration["parameters"],
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
    _iterator = _SKOptIter
    _str_attr = ["search_space", "n_iter", "objective"]

    def __init__(self, search_space, n_iter, objective, **kwargs):
        self.search_space = search_space
        self.n_iter = n_iter
        self.objective = objective
        self.kwargs = kwargs

    @property
    def varying_parameters(self) -> Mapping:
        """Parameters in this sampler. Does not include parameters that do not vary."""
        return self.search_space
