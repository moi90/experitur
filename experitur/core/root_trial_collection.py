import hashlib
import itertools
from typing import TYPE_CHECKING, List, Mapping

from experitur.core.trial import BaseTrialCollection, Trial, TrialCollection
from experitur.core.trial_store import KeyExistsError
from experitur.helpers.merge_dicts import merge_dicts
from experitur.util import callable_to_name

if TYPE_CHECKING:
    from experitur.core.context import Context


class RootTrialCollection(BaseTrialCollection):
    def __init__(self, ctx: "Context"):
        BaseTrialCollection.__init__(self)

        self.ctx = ctx

    def update(self, trial: Trial):
        self.ctx.store[trial.id] = trial._data

    def remove(self, trial: Trial):
        self.ctx.store.delete(trial.id, trial.revision)

    # __contains__, __iter__, __len__

    def __contains__(self, trial: Trial):
        raise NotImplementedError()

    def __iter__(self):
        for trial_data in self.ctx.store.values():
            yield Trial(trial_data, self)

    def __len__(self):
        return len(self.ctx.store)

    def get(self, trial_id):
        trial_data = self.ctx.store[trial_id]
        return Trial(trial_data, self)

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> TrialCollection:
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            if experiment.name is None:
                raise ValueError(f"Experiment {experiment!r} has no name set")
            experiment = experiment.name

        from experitur.core.trial_store import _match_parameters

        trials = [
            Trial(td, self)
            for td in self.ctx.store.match(
                func, parameters, experiment, resolved_parameters
            )
        ]

        return TrialCollection(trials)

    def _create(self, trial_data) -> dict:
        trial_data.setdefault("parameters", {})

        trial_parameters = trial_data["parameters"]
        experiment_varying_parameters = sorted(
            trial_data["experiment"]["varying_parameters"]
        )
        experiment_name = trial_data["experiment"]["name"]

        # First try: Use the varying parameters of the currently running experiment
        trial_id = _format_trial_id(
            experiment_name, trial_parameters, experiment_varying_parameters
        )

        trial_data["wdir"] = self.ctx.get_trial_wdir(trial_id)

        try:
            return self.ctx.store.create(trial_id, trial_data)
        except KeyExistsError:
            pass

        existing_trial_data = self.ctx.store[trial_id]

        # Second try: Incorporate more independent parameters
        new_independent_parameters = []

        # Look for parameters in existing_trial that have differing values
        for name, value in existing_trial_data["parameters"].items():
            if name in trial_parameters and trial_parameters[name] != value:
                new_independent_parameters.append(name)

        # Look for parameters that did not exist previously
        for name in trial_parameters.keys():
            if name not in existing_trial_data["parameters"]:
                new_independent_parameters.append(name)

        if new_independent_parameters:
            trial_id = _format_trial_id(
                experiment_name,
                trial_parameters,
                sorted(set(experiment_varying_parameters + new_independent_parameters)),
            )

            trial_data["wdir"] = self.ctx.get_trial_wdir(trial_id)

            try:
                return self.ctx.store.create(trial_id, trial_data)
            except KeyExistsError:
                pass

        # Otherwise, just append a version number
        for i in itertools.count(1):
            test_trial_id = "{}.{}".format(trial_id, i)

            trial_data["wdir"] = self.ctx.get_trial_wdir(test_trial_id)

            try:
                return self.ctx.store.create(test_trial_id, trial_data)
            except KeyExistsError:
                continue

    def create(self, trial_data, **kwargs) -> Trial:
        """Create a :py:class:`Trial` instance."""

        # Initialize defaults
        trial_data = merge_dicts(
            {"parameters": {}, "resolved_parameters": {}}, trial_data
        )

        trial_data = self._create(trial_data)

        assert "wdir" in trial_data

        return Trial(trial_data, self, **kwargs)


def _format_trial_id(
    experiment_name,
    trial_parameters: Mapping,
    independent_parameters: List[str],
    shorten=True,
):
    parameter_values = sorted(
        (
            "{}-{!s}".format(k, trial_parameters[k])
            for k in independent_parameters
            if k in trial_parameters
        ),
        key=len,
    )

    if len(parameter_values) > 0:
        hashed = []
        while True:
            hashed_str = hashlib.sha1("".join(hashed).encode()).hexdigest()[:7]
            parts = ([hashed_str] if hashed else []) + sorted(parameter_values)
            trial_id = "_".join(parts)

            if not shorten or len(trial_id) < 192 or not parameter_values:
                break

            hashed.append(parameter_values.pop())

        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return f"{experiment_name}/{trial_id}"
