from experitur.recursive_formatter import RecursiveDict
from sklearn.model_selection import ParameterGrid
import yaml
from functools import reduce
from datetime import datetime
from random import shuffle
from etaprogress.progress import ProgressBar
from timer_cm import Timer


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

        return cfg

    def run(self):
        if isinstance(self.configuration, list):
            for exp in self.configuration:
                self._run_single(exp)
        elif isinstance(self.configuration, dict):
            self._run_single(self.configuration)
        else:
            raise ExperimentError(
                "Configuration is expected to consist of a list or a dict!")

    def _run_single(self, configuration):
        configuration.setdefault("parameters", {})

        independent_parameters = sorted(
            k for k, v in configuration["parameters"].items() if len(v) > 1)

        print("Independent parameters:", independent_parameters)

        parameter_grid = list(ParameterGrid(configuration["parameters"]))

        if configuration.get("shuffle_experiments", False):
            shuffle(parameter_grid)

        bar = ProgressBar(len(parameter_grid), max_width=40)

        with Timer("Overall") as timer:
            for i, p in enumerate(parameter_grid):
                ident = "_".join("{}-{!s}".format(k, p[k])
                                 for k in independent_parameters)
                ident = ident.replace("/", "_")

                print("Experiment {}: {}".format(i, ident))
                print(bar)

                p = RecursiveDict(p, allow_missing=True)
                for k, v in sorted(p.items()):
                    print("    {}: {}".format(k, v))

                with timer.child(ident):
                    # Run experiment
                    ...

                bar.numerator += 1
