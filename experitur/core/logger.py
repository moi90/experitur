import collections.abc
import os.path
import weakref
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import cachetools
import yaml

from experitur.helpers.dumper import ExperiturDumper

if TYPE_CHECKING:
    from experitur.core.trial import Trial

try:
    import pandas as pd
except ImportError:
    pd = None

LogValues = Mapping[str, Any]


class LoggerBase(ABC):
    def __init__(self, trial: "Trial"):
        self.trial = weakref.proxy(trial)  # Avoid cycles
        self.last_entry = {}

    def log(self, values: LogValues):
        self.last_entry.update(values)
        self._log(values)

    @abstractmethod
    def _log(self, values: LogValues):
        pass

    @abstractmethod
    def read(self) -> List[LogValues]:
        pass

    def to_pandas(self):
        if pd is None:
            raise ImportError(name="pandas")

        return pd.json_normalize(self.read())

    def aggregate(self, include) -> Dict:
        include = set(include)

        metrics = defaultdict(list)
        for entry in self.read():
            for k, v in entry.items():
                if k in include:
                    metrics[k].append(v)

        result = {}
        for k in metrics:
            result[f"max_{k}"] = max(metrics[k])
            result[f"min_{k}"] = min(metrics[k])
            result[f"final_{k}"] = metrics[k][-1]
            result[f"mean_{k}"] = sum(metrics[k]) / len(metrics[k])

        return result


class YAMLLogger(LoggerBase):
    # Dict[filename, Tuple[mtime, list]]
    _cache = cachetools.LRUCache(128)

    def __init__(self, trial: "Trial"):
        super().__init__(trial)

        self.log_fn = os.path.join(self.trial.wdir, "log.yaml")

    def _log(self, values: LogValues):
        if not isinstance(values, collections.abc.Mapping):
            raise ValueError(f"Expected mapping, got {values!r}")

        with open(self.log_fn, "a") as fp:
            yaml.dump(values, fp, Dumper=ExperiturDumper, explicit_start=True)

    def read(self):
        try:
            mtime = os.path.getmtime(self.log_fn)
        except FileNotFoundError:
            return []

        try:
            cache_entry = self._cache[self.log_fn]
        except KeyError:
            pass
        else:
            if cache_entry[0] == mtime:
                return cache_entry[1]

        try:
            with open(self.log_fn, "r") as fp:
                log = list(yaml.load_all(fp, Loader=yaml.Loader))
        except FileNotFoundError:
            return []

        self._cache[self.log_fn] = (mtime, log)

        return log
