from experitur.context import Context, experiment, push_context, run
from experitur.errors import ExperiturError

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
