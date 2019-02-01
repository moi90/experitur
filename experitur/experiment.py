import copy
import datetime
import errno
import os
import shutil
import sys
import traceback
from importlib import import_module
from random import shuffle

import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid
from timer_cm import Timer
from tqdm import tqdm

from experitur.backends import FileBackend
from experitur.helpers.dumper import ExperiturDumper
from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict


class ExperimentError(Exception):
    pass


def _check_parameter_grid(parameter_grid):
    """
    parameter_grid has to be a dict of lists.
    """

    if not isinstance(parameter_grid, dict):
        raise ExperimentError("parameter_grid is expected to be a dictionary.")

    errors = []
    for k, v in parameter_grid.items():
        if not isinstance(v, list):
            errors.append(k)

    if errors:
        raise ExperimentError(
            "Parameters {} are not lists.".format(", ".join(errors)))


class Experiment:
    def __init__(self, filename, experiment_root=None):
        self.configuration = self._load(filename)
        self.filename = filename

        if experiment_root is None:
            self.experiment_root = os.path.splitext(self.filename)[0]
        else:
            self.experiment_root = experiment_root

    def _load(self, filename):
        with open(filename) as f:
            try:
                cfg = next(yaml.safe_load_all(f))
            except yaml.YAMLError as e:
                raise ExperimentError(
                    "{} contains a malformed YAML document!".format(filename)) from e
            except StopIteration as e:
                raise ExperimentError(
                    "{} contains no YAML documents!".format(filename)) from e

            if cfg is None:
                raise ExperimentError(
                    "{} contains an empty configuration!".format(filename))

        if not isinstance(cfg, (list, dict)):
            raise ExperimentError(
                "Configuration is expected to consist of a list or a dict!")

        if not isinstance(cfg, list):
            cfg = [cfg]

        return cfg

    def _make_backend(self, experiment_id):
        experiment_path = os.path.join(
            self.experiment_root,
            experiment_id)
        return FileBackend(experiment_path)

    def run(self, skip_existing=True, halt_on_error=True):
        results = []

        # TODO: Use a contextmanager for this!
        # Add experiment directory to sys.path
        root = os.path.abspath(os.path.dirname(self.filename))
        sys.path.insert(0, root)
        for i, exp_config in enumerate(self.configuration):
            # Fill in data from base experiments
            exp_config = self._merge_base_experiment(exp_config)

            exp_config.setdefault("id", "_{}".format(i))

            if "run" not in exp_config:
                print("Experiment {} is abstract. Skipping.".format(
                    exp_config["id"]))
                continue

            results.extend(
                self._run_single(
                    exp_config, skip_existing=skip_existing, halt_on_error=halt_on_error))

        return results

    def _merge_base_experiment(self, exp_config):
        try:
            base_id = exp_config["base"]
        except KeyError:
            return exp_config

        base = None
        for candidate in self.configuration:
            if candidate.get("id") == base_id:
                base = candidate

        if base is None:
            raise ExperimentError("Base ID {} not found!".format(base_id))

        # Copy base and exp_config so nothing gets overwritten
        base, exp_config = copy.deepcopy(base), copy.deepcopy(exp_config)
        del base["id"]
        del exp_config["base"]

        merged = merge_dicts(dict(base), exp_config)

        if "base" in merged:
            # Recurse for multiple inheritance
            return self._merge_base_experiment(merged)

        return merged

    def _run_single(self, exp_config, skip_existing=True, halt_on_error=True):
        print("Running experiment {}...".format(exp_config["id"]))

        exp_config.setdefault("parameter_grid", {})

        _check_parameter_grid(exp_config["parameter_grid"])

        independent_parameters = sorted(
            k for k, v in exp_config["parameter_grid"].items() if len(v) > 1)

        print("Independent parameters:", independent_parameters)

        parameter_grid = list(ParameterGrid(exp_config["parameter_grid"]))

        if exp_config.get("shuffle_trials", False):
            print("Trials are shuffled.")
            shuffle(parameter_grid)

        run_module_name, run_function_name = exp_config["run"].split(":", 1)

        try:
            run_module = import_module(run_module_name)
        except ImportError as exc:
            raise ExperimentError(
                "Error loading {}!".format(run_module_name)) from exc

        try:
            run = getattr(run_module, run_function_name)
        except AttributeError as exc:
            print(dir(run_module))
            raise ExperimentError(
                "Run function {}:{} not found!".format(run_module_name, run_function_name)) from exc

        # Create a working directory for this experiment
        experiment_path = os.path.join(
            self.experiment_root,
            exp_config["id"])

        os.makedirs(experiment_path, exist_ok=True)

        backend = self._make_backend(exp_config["id"])

        #bar = ProgressBar(len(parameter_grid), max_width=40)
        pbar = tqdm(total=len(parameter_grid), unit="")

        results = []

        with Timer("Total time") as timer:
            for i, trial_parameters in enumerate(parameter_grid):
                trial_parameters = RecursiveDict(
                    trial_parameters, allow_missing=True).as_dict()

                # Check, if a trial with this parameter set already exists
                existing = backend.find_trials_by_parameters(trial_parameters)

                if skip_existing and len(existing):
                    print("Skip existing trial: {}".format(
                        backend.format_independent_parameters(trial_parameters, independent_parameters)))
                    continue

                trial_id = backend.make_trial_id(
                    trial_parameters, independent_parameters)

                trial_dir = os.path.join(experiment_path, trial_id)

                # Create directory
                os.makedirs(trial_dir, exist_ok=True)

                pbar.set_description(
                    "Trial {}".format(trial_id), refresh=True)
                pbar.update()

                for k, v in sorted(trial_parameters.items()):
                    print("    {}: {}".format(k, v))

                with timer.child(trial_id):
                    result = None
                    trial_data = {
                        "parameters_pre": copy.deepcopy(trial_parameters),
                        "success": False,
                        "time_start": datetime.datetime.now(),
                        "experiment_id": exp_config.get("id"),
                        "trial_id": trial_id,
                    }

                    # Run experiment
                    try:
                        result = run(trial_dir, trial_parameters)
                    except (Exception, KeyboardInterrupt) as exc:
                        # TODO: Log e
                        traceback.print_exc()

                        trial_data["error"] = ": ".join(
                            filter(None, (exc.__class__.__name__, str(exc))))

                        # TODO: Halt further execution on exception
                        if isinstance(exc, KeyboardInterrupt) or halt_on_error:
                            # or exp_config.get("raise_exceptions", True):
                            raise exc
                    else:
                        trial_data["success"] = True
                    finally:
                        trial_data["result"] = result
                        trial_data["time_end"] = datetime.datetime.now()
                        trial_data["parameters_post"] = trial_parameters

                        with open(os.path.join(trial_dir, "experitur.yaml"), "w") as fp:
                            yaml.dump(trial_data, fp, Dumper=ExperiturDumper)

                        results.append(result)

        pbar.close()
        print()

        return results

    def _clean_experiment(self, experiment_root, experiment_id, dry_run=False, failed=False):
        backend = self._make_backend(experiment_id)
        for trial_id, trial_data in backend.trials():
            trial_dir = os.path.join(
                experiment_root, experiment_id, trial_id)
            if failed and (trial_data.get("error", False)
                           or not trial_data.get("success", True)):
                print("Removing failed {}/{}...".format(experiment_id, trial_id))
                if not dry_run:
                    shutil.rmtree(trial_dir)

        if not dry_run:
            # Don't leave empty directories behind
            try:
                os.removedirs(os.path.join(experiment_root, experiment_id))
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise

    # TODO: Support empty, successful
    def clean(self, experiment_id=None, **kwargs):
        if not os.path.isdir(self.experiment_root):
            # There's nothing to do
            return

        if experiment_id is not None:
            self._clean_experiment(
                self.experiment_root, experiment_id, **kwargs)
        else:
            for dirent in os.scandir(self.experiment_root):
                if not dirent.is_dir():
                    continue

                experiment_id = dirent.name
                self._clean_experiment(
                    self.experiment_root, experiment_id, **kwargs)

        # if remove_everything:
        #     shutil.rmtree(experiment_root)

        # else:
        #     for trial_dir in glob.iglob("{}/*/*/".format(experiment_root)):
        #         contents = os.listdir(trial_dir)

        #         if not contents:
        #             print(trial_dir)
        #             os.removedirs(trial_dir)

    def collect(self, failed=False):
        data = {}
        for dirent in os.scandir(self.experiment_root):
            if not dirent.is_dir():
                continue
            experiment_id = dirent.name
            backend = self._make_backend(experiment_id)
            for trial_id, trial_data in backend.trials():
                if not failed and (
                        trial_data.get("error", False)
                        or not trial_data.get("success", True)):
                    print("Skipping failed {}/{}...".format(experiment_id, trial_id))
                    continue

                data["{}/{}".format(experiment_id, trial_id)] = trial_data

        data = {k: _extract_results(v) for k, v in data.items()}
        data = pd.DataFrame.from_dict(data, orient="index")
        data.index.name = "trial_id"

        # TODO: Remove columns that are not serializable in CSV

        result_fn = os.path.join(self.experiment_root, "results.csv")
        data.to_csv(result_fn)


def _extract_results(trial):
    result = {}

    for k, v in trial.items():
        if k in ("parameters_post", "parameters_pre", "result", "trial_id"):
            continue
        result["meta_{}".format(k)] = v

    for k, v in trial.get("parameters_post", {}).items():
        result["{}".format(k)] = v

    trial_result = trial.get("result", {})

    if isinstance(trial_result, dict):
        for k, v in trial_result.items():
            result["{}_".format(k)] = v
    else:
        result["result_"] = trial_result

    return result
