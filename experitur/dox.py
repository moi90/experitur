import importlib.util
import os.path
import sys

from experitur.errors import ExperiturError
from experitur.core.experiment import Experiment


class DOXError(ExperiturError):
    pass


def load_dox(dox_fn):
    name, ext = os.path.splitext(os.path.basename(dox_fn))

    try:
        loader = _LOADERS[ext]
    except KeyError as exc:
        msg = "Unrecognized file extension: {}. Use {}.".format(
            ext, ", ".join(_LOADERS.keys())
        )
        raise DOXError(msg) from exc

    return loader(dox_fn, name)


def _load_py(dox_fn, dox_name):
    # Insert the location of dox_fn to the sys path so that DOXes can import stuff
    sys.path.insert(0, os.path.abspath(os.path.dirname(dox_fn)))

    try:
        spec = importlib.util.spec_from_file_location(dox_name, dox_fn)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DOXError("Error loading {}!".format(dox_fn)) from exc

    # Insert into sys.modules
    sys.modules[module.__name__] = module

    # Guess experiment names from variable name in module
    for name in dir(module):
        experiment: Experiment = getattr(module, name)
        if not isinstance(experiment, Experiment):
            continue

        if not experiment.name:
            experiment.name = name


_LOADERS = {".py": _load_py}
