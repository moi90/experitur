import collections.abc
import os.path
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import cachetools

from experitur.helpers import yaml

if TYPE_CHECKING:
    from experitur.core.trial import Trial

try:
    import pandas as pd
except ImportError:
    pd = None

LogValues = Mapping[str, Any]


class Message(Enum):
    COMMIT = "commit"
    ROLLBACK = "rollback"


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

    def aggregate(self, include=None, exclude=None) -> Dict:
        """
        Aggregate (min/max/mean/final) each field in the log.

        Args:
            include (Collection, optional): If not None, include only these fields.
            exclude (Collection, optional): If not None, exclude these fields.
        """

        include = set(include) if include is not None else None
        exclude = set(exclude) if exclude is not None else None

        metrics = defaultdict(list)
        for entry in self.read():
            for k, v in entry.items():
                if include is not None and k not in include:
                    continue

                if exclude is not None and k in exclude:
                    continue

                metrics[k].append(v)

        result = {}
        for k in metrics:
            result[f"max_{k}"] = max(metrics[k])
            result[f"min_{k}"] = min(metrics[k])
            result[f"mean_{k}"] = sum(metrics[k]) / len(metrics[k])
            result[f"final_{k}"] = metrics[k][-1]

        return result

    @abstractmethod
    def commit(self) -> None:
        """
        Commit recently written entries.

        Called by Trial.save_checkpoint.
        """

    @abstractmethod
    def rollback(self) -> None:
        """
        Rollback uncommitted entries.

        Called by Trial.load_checkpoint.
        """


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
            yaml.dump(values, fp, Dumper=yaml.Dumper, explicit_start=True)

    def _read(self) -> List:
        result = []
        queue = []
        with open(self.log_fn, "r") as fp:
            for entry in yaml.load_all(fp, Loader=yaml.Loader):
                if entry is Message.ROLLBACK:
                    queue.clear()
                elif entry is Message.COMMIT:
                    result.extend(queue)
                    queue.clear()
                else:
                    queue.append(entry)

        # Append entries that were not rolled back
        result.extend(queue)
        return result

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
            log = self._read()
        except FileNotFoundError:
            return []

        self._cache[self.log_fn] = (mtime, log)

        return log

    def commit(self):
        with open(self.log_fn, "a") as fp:
            yaml.dump(Message.COMMIT, fp, Dumper=yaml.Dumper, explicit_start=True)

    def rollback(self):
        with open(self.log_fn, "a") as fp:
            yaml.dump(Message.ROLLBACK, fp, Dumper=yaml.Dumper, explicit_start=True)
