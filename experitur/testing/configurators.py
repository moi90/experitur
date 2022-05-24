from experitur.helpers.merge_dicts import merge_dicts
from typing import Mapping, Optional
import itertools
from experitur.core.configurators import BaseConfigurationSampler


def all_parameter_subsets(configuration: Mapping):
    """Generate all parameter subsets of a configuration to test Sampler.contains_superset_of."""
    parameters = configuration.get("parameters", {})

    key_sets = itertools.chain.from_iterable(
        itertools.combinations(parameters.keys(), l) for l in range(len(parameters) + 1)
    )

    return [
        dict(configuration, parameters={k: parameters[k] for k in keys})
        for keys in key_sets
    ]


def assert_sampler_contains_subset_of_all_samples(
    sampler: BaseConfigurationSampler, include_parameters: Optional[Mapping] = None
):
    """
    Assert that sampler.contains_subset_of is true for all generated samples.

    Args:
        sampler (BaseConfigurationSampler): Sampler.
        include_parameters (Mapping, optional): Additional parameters to test if
            sampler.contains_subset_of also works for strict subsets.
    """
    for conf in sampler:
        assert sampler.contains_subset_of(
            conf
        ), f"No subset of {conf} in sampler {sampler}"

        if include_parameters is not None:
            superset = merge_dicts(conf, parameters=include_parameters)
            assert sampler.contains_subset_of(
                superset
            ), f"No subset of {superset} in sampler"


def assert_sampler_contains_superset_of_all_samples(
    sampler: BaseConfigurationSampler, with_subsets: bool = True
):
    """
    Assert that sampler.contains_superset_of is true for all generated samples (and their subsets).

    Args:
        sampler (BaseConfigurationSampler): Sampler.
        include_subsets (bool, optional): Test all subsets of a configuration.
    """

    for conf in sampler:
        if with_subsets:
            for subset in all_parameter_subsets(conf):
                assert sampler.contains_superset_of(
                    subset
                ), f"No superset of {subset} in sampler"
        else:
            assert sampler.contains_superset_of(
                conf
            ), f"No superset of {conf} in sampler"
