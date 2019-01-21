import shutil
import errno
import glob
import os
import pprint
from datetime import datetime
from functools import reduce
from importlib import import_module
from random import shuffle
import json

import yaml
from etaprogress.progress import ProgressBar
from sklearn.model_selection import ParameterGrid
from timer_cm import Timer

from experitur.helpers.merge_dicts import merge_dicts
from experitur.recursive_formatter import RecursiveDict

import pickle
import copy

import zipfile


class ExperimentError(Exception):
    pass


class Experiment:
    def __init__(self, filename):
        self.configuration = self._load(filename)
        self.filename = filename

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

    def run(self):
        results = []
        for i, exp_config in enumerate(self.configuration):
            # Fill in data from base experiments
            exp_config = self._merge_base_experiment(exp_config)

            exp_config.setdefault("id", "_{}".format(i))

            if "run" not in exp_config:
                print("Experiment {} is abstract. Skipping.".format(
                    exp_config["id"]))
                continue

            results.extend(self._run_single(exp_config))

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

    def _run_single(self, exp_config):
        exp_config.setdefault("parameter_grid", {})

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
        except ModuleNotFoundError as e:
            raise ExperimentError(
                "Error loading {}!".format(run_module_name)) from e

        try:
            run = getattr(run_module, run_function_name)
        except AttributeError as e:
            print(dir(run_module))
            raise ExperimentError(
                "Run function {}:{} not found!".format(run_module_name, run_function_name)) from e

        # Create a working directory for this experiment
        experiment_root = os.path.join(
            os.path.splitext(self.filename)[0],
            exp_config["id"])

        os.makedirs(experiment_root, exist_ok=True)

        bar = ProgressBar(len(parameter_grid), max_width=40)

        results = []

        with Timer("Overall") as timer:
            for i, p in enumerate(parameter_grid):
                if len(independent_parameters) > 0:
                    ident = "_".join("{}-{!s}".format(k, p[k])
                                     for k in independent_parameters)
                    ident = ident.replace("/", "_")
                else:
                    ident = "_"

                trial_dir = os.path.join(experiment_root, ident)

                try:
                    os.mkdir(trial_dir)
                except OSError as exc:
                    if exc.errno == errno.EEXIST:
                        print("Skipping {}, directory already exists: {}".format(
                            ident, trial_dir))
                        continue
                    else:
                        raise

                print("Trial {}: {}".format(i, ident))
                print(bar)

                p = RecursiveDict(p, allow_missing=True).as_dict()

                for k, v in sorted(p.items()):
                    print("    {}: {}".format(k, v))

                with timer.child(ident):
                    trial_data = {}
                    trial_data["parameters_pre"] = copy.deepcopy(p)
                    trial_data["success"] = False

                    result = None

                    # Run experiment
                    try:
                        result = run(working_directory=trial_dir, parameters=p)
                    except (Exception, KeyboardInterrupt) as exc:
                        # TODO: Log e
                        print(exc)

                        trial_data["error"] = ": ".join(
                            filter(None, (exc.__class__.__name__, str(exc))))

                        if isinstance(exc, KeyboardInterrupt) or exp_config.get("raise_exceptions", True):
                            raise exc
                    else:
                        trial_data["success"] = True
                    finally:
                        trial_data["result"] = result
                        trial_data["parameters_post"] = p

                        with open(os.path.join(trial_dir, "experitur.yaml"), "w") as fp:
                            yaml.dump(trial_data, fp)

                        results.append(result)

                bar.numerator += 1

        return results

    def clean(self, remove_everything=False):
        experiment_root = os.path.splitext(self.filename)[0]

        if remove_everything:
            shutil.rmtree(experiment_root)

        else:
            for trial_dir in glob.iglob("{}/*/*/".format(experiment_root)):
                contents = os.listdir(trial_dir)

                if not contents:
                    print(trial_dir)
                    os.removedirs(trial_dir)
