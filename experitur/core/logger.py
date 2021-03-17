import os.path
import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Mapping
import collections.abc

import yaml

from experitur.helpers.dumper import ExperiturDumper

if TYPE_CHECKING:
    from experitur.core.trial import Trial

try:
    import pandas as pd
except ImportError:
    pd = None

LogValues = Mapping[str, Any]


class LoggerBase:
    def __init__(self, trial: "Trial"):
        self.trial = weakref.proxy(trial)  # Avoid cycles

    @abstractmethod
    def log(self, values: LogValues):
        pass

    @abstractmethod
    def read(self) -> List[LogValues]:
        pass

    def to_pandas(self):
        if pd is None:
            raise ImportError(name="pandas")

        return pd.json_normalize(self.read())


class YAMLLogger(LoggerBase):
    def __init__(self, trial: "Trial"):
        super().__init__(trial)

        self.log_fn = os.path.join(self.trial.wdir, "log.yaml")

    def log(self, values: Mapping):
        if not isinstance(values, collections.abc.Mapping):
            raise ValueError(f"Expected mapping, got {values!r}")

        with open(self.log_fn, "a") as fp:
            yaml.dump(values, fp, Dumper=ExperiturDumper, explicit_start=True)

    def read(self):
        with open(self.log_fn, "r") as fp:
            return list(yaml.load_all(fp, Loader=yaml.Loader))
