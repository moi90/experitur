from experitur.recursive_formatter import RecursiveDict
from sklearn.model_selection import ParameterGrid
import yaml
from functools import reduce
from datetime import datetime
from random import shuffle
from etaprogress.progress import ProgressBar
from timer_cm import Timer
from experitur.helpers.merge_dicts import merge_dicts
from importlib import import_module


class ExperimentError(Exception):
    pass


class Experiment:
    def __init__(self, filename):
        self.configuration = self._load(filename)

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
        for exp_config in self.configuration:
            # Fill in data from base experiments
            exp_config = self._merge_base_experiment(exp_config)

            self._run_single(exp_config)

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

        # Copy base and exp_config
        base, exp_config = dict(base), dict(exp_config)
        del base["id"]
        del exp_config["base"]

        merged = merge_dicts(dict(base), exp_config)

        if "base" in merged:
            # Recurse for multiple inheritance
            return self._merge_base_experiment(merged)

        return merged

    def _run_single(self, exp_config):
        exp_config.setdefault("parameters", {})

        independent_parameters = sorted(
            k for k, v in exp_config["parameters"].items() if len(v) > 1)

        print("Independent parameters:", independent_parameters)

        parameter_grid = list(ParameterGrid(exp_config["parameters"]))

        if exp_config.get("shuffle_trials", False):
            print("Trials are shuffled.")
            shuffle(parameter_grid)

        run_module_name, run_function_name = exp_config["run"].split(":", 1)

        try:
            run_module = import_module(run_module_name)
        except ModuleNotFoundError as e:
            raise ExperimentError(
                "Run module {} not found!".format(run_module_name)) from e

        try:
            run = getattr(run_module, run_function_name)
        except AttributeError as e:
            print(dir(run_module))
            raise ExperimentError(
                "Run function {}:{} not found!".format(run_module_name, run_function_name)) from e

        bar = ProgressBar(len(parameter_grid), max_width=40)

        with Timer("Overall") as timer:
            for i, p in enumerate(parameter_grid):
                ident = "_".join("{}-{!s}".format(k, p[k])
                                 for k in independent_parameters)
                ident = ident.replace("/", "_")

                print("Trial {}: {}".format(i, ident))
                print(bar)

                p = RecursiveDict(p, allow_missing=True)
                for k, v in sorted(p.items()):
                    print("    {}: {}".format(k, v))

                with timer.child(ident):
                    # Run experiment
                    run(**p)

                bar.numerator += 1
