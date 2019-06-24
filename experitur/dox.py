import importlib.util
import os.path
import sys
from importlib import import_module

from experitur.context import Context, push_context
from experitur.errors import ExperiturError


class DOXError(ExperiturError):
    pass


def load_dox(dox_fn):
    name, ext = os.path.splitext(os.path.basename(dox_fn))

    try:
        loader = _LOADERS[ext]
    except KeyError as exc:
        msg = "Unrecognized file extension: {}. Use {}.".format(
            ext, ", ".join(_LOADERS.keys()))
        raise DOXError(msg) from exc

    return loader(dox_fn, name)


def _load_py(dox_fn, dox_name):
    # Insert the location of dox_fn to the sys path so that DOXes can import stuff
    sys.path.insert(0, os.path.abspath(os.path.dirname(dox_fn)))

    try:
        spec = importlib.util.spec_from_file_location(
            dox_name, dox_fn)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DOXError(
            "Error loading {}!".format(dox_fn)) from exc


_LOADERS = {
    ".py": _load_py
}
