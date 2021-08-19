from experitur.access import get_trial, get_current_trial
from experitur.core.experiment import Experiment
from experitur.core.trial import Trial
from experitur.errors import ExperiturError
from experitur.core.context import Context, get_current_context

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
