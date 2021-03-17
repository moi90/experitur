import collections.abc
import glob
import itertools
import os.path
import shutil
import typing
from abc import abstractmethod
from typing import Dict, List, Mapping

import yaml

from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict
from experitur.util import callable_to_name

if typing.TYPE_CHECKING:
    from experitur.core.context import Context


def _match_parameters(parameters_1, parameters_2):
    """Decide whether parameters_1 are a subset of parameters_2."""

    if set(parameters_1.keys()) <= set(parameters_2.keys()):
        return all(v == parameters_2[k] for k, v in parameters_1.items())

    return False


def _format_trial_id(
    experiment_name, trial_parameters: Mapping, independent_parameters: List[str]
):
    if len(independent_parameters) > 0:
        trial_id = "_".join(
            "{}-{!s}".format(k, trial_parameters.get(k, "na"))
            for k in independent_parameters
        )
        trial_id = trial_id.replace("/", "_")
    else:
        trial_id = "_"

    return f"{experiment_name}/{trial_id}"


class KeyExistsError(Exception):
    pass


class TrialStore(collections.abc.MutableMapping):
    def __init__(self, ctx: "Context"):
        self.ctx = ctx

    def match(
        self, func=None, parameters=None, experiment=None, resolved_parameters=None
    ) -> List[Dict]:
        func = callable_to_name(func)

        from experitur.core.experiment import Experiment

        if isinstance(experiment, Experiment):
            experiment = experiment.name

        trial_data_list = []
        for trial in self.values():
            experiment_ = trial.get("experiment", {})
            if func is not None and callable_to_name(experiment_.get("func")) != func:
                continue

            if parameters is not None and not _match_parameters(
                parameters, trial.get("parameters", {})
            ):
                continue

            if resolved_parameters is not None and not _match_parameters(
                resolved_parameters, trial.get("resolved_parameters", {})
            ):
                continue

            if experiment is not None and experiment_.get("name") != str(experiment):
                continue

            trial_data_list.append(trial)

        return trial_data_list

    @abstractmethod
    def _create(self, trial_data):
        """
        Create an entry using the trial_data["id"].

        Returns:
            trial_data

        Raises:
            KeyExistsError if a trial with the specified id already exists.
        """

    def create(self, trial_configuration):
        """Create a :py:class:`Trial` instance."""

        trial_configuration.setdefault("parameters", {})

        trial_configuration = merge_dicts(
            trial_configuration,
            resolved_parameters=RecursiveDict(
                trial_configuration["parameters"], allow_missing=True
            ).as_dict(),
        )

        trial_parameters = trial_configuration["parameters"]
        experiment_varying_parameters = sorted(
            trial_configuration["experiment"]["varying_parameters"]
        )
        experiment_name = trial_configuration["experiment"]["name"]

        # First try: Use the varying parameters of the currently running experiment
        trial_id = _format_trial_id(
            experiment_name, trial_parameters, experiment_varying_parameters
        )

        try:
            return self._create(merge_dicts(trial_configuration, id=trial_id))
        except KeyExistsError:
            pass

        existing_trial = self[trial_id]

        # Second try: Incorporate more independent parameters
        new_independent_parameters = []

        # Look for parameters in existing_trial that have differing values
        for name, value in existing_trial["parameters"].items():
            if name in trial_parameters and trial_parameters[name] != value:
                new_independent_parameters.append(name)

        # Look for parameters that did not exist previously
        for name in trial_parameters.keys():
            if name not in existing_trial["parameters"]:
                new_independent_parameters.append(name)

        if new_independent_parameters:

            trial_id = _format_trial_id(
                experiment_name,
                trial_parameters,
                sorted(set(experiment_varying_parameters + new_independent_parameters)),
            )

            try:
                return self._create(merge_dicts(trial_configuration, id=trial_id))
            except KeyExistsError:
                pass

        # Otherwise, just append a version number
        for i in itertools.count(1):
            test_trial_id = "{}.{}".format(trial_id, i)

            try:
                return self._create(merge_dicts(trial_configuration, id=test_trial_id))
            except KeyExistsError:
                continue

    def check_writable(self):
        __tracebackhide__ = True  # pylint: disable=unused-variable

        if not self.ctx.writable:
            raise RuntimeError("Context is not writable.")

    def delete_all(self, keys):
        self.check_writable()

        for k in keys:
            del self[k]


class FileTrialStore(TrialStore):
    TRIAL_FN = "trial.yaml"
    DUMPER = ExperiturDumper

    def _get_trial_fn(self, trial_id):
        return os.path.join(self.ctx.get_trial_wdir(trial_id), self.TRIAL_FN)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, trial_id):
        path = self._get_trial_fn(trial_id)

        try:
            with open(path) as fp:
                return yaml.load(fp, Loader=yaml.Loader)
        except FileNotFoundError as exc:
            raise KeyError(trial_id) from exc
        except:
            print(f"Error reading {path}")
            raise

    def __iter__(self):
        path = self._get_trial_fn("**")

        left, right = path.split("**", 1)

        for entry_fn in glob.iglob(path, recursive=True):
            if os.path.isdir(entry_fn):
                continue

            # Convert entry_fn back to key
            k = entry_fn[len(left) : -len(right)]

            # Keys use forward slashes
            k = k.replace("\\", "/")

            yield k

    def _create(self, trial_data: dict):
        trial_id = trial_data["id"]

        if not isinstance(trial_data, dict):
            raise ValueError(f"trial_data has to be dict, got {trial_data!r}")

        wdir = self.ctx.get_trial_wdir(trial_id)

        try:
            os.makedirs(wdir, exist_ok=False)
        except FileExistsError:
            raise KeyExistsError(trial_id) from None

        path = self._get_trial_fn(trial_id)
        with open(path, "x") as fp:
            yaml.dump(trial_data, fp, Dumper=self.DUMPER)

        return trial_data

    def __setitem__(self, trial_id: str, trial_data: dict):
        self.check_writable()

        if not isinstance(trial_data, dict):
            raise ValueError(f"trial_data has to be dict, got {trial_data!r}")

        path = self._get_trial_fn(trial_id)

        try:
            with open(path, "r+") as fp:
                fp.truncate()
                yaml.dump(trial_data, fp, Dumper=self.DUMPER)
        except FileNotFoundError:
            raise KeyError(trial_id) from None

    def __delitem__(self, trial_id):
        self.check_writable()

        path = self._get_trial_fn(trial_id)

        try:
            os.remove(path)
        except FileNotFoundError:
            raise KeyError

        shutil.rmtree(os.path.dirname(path))
