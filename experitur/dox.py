import importlib.util
import os.path
from importlib import import_module

from experitur.context import Context, push_context
from experitur.errors import ExperiturError


class DOXError(ExperiturError):
    pass


class DOX:
    def __init__(self, dox_fn, wdir=None):
        wdir_, ext = os.path.splitext(dox_fn)
        dox_name = os.path.basename(wdir_)

        if wdir is None:
            wdir = wdir_

        os.makedirs(wdir, exist_ok=True)

        self.ctx = Context(wdir=wdir)

        if ext == ".py":
            # Python code path
            self._load_py(dox_fn, dox_name)
        elif ext in (".md", ".yaml"):
            # YAML code path
            ...
        else:
            msg = "Unrecognized file extension: {}. Use .py, .yaml or .md.".format(
                ext)
            raise DOXError(msg)

    def _load_py(self, dox_fn, dox_name):
        try:
            with push_context(self.ctx):
                spec = importlib.util.spec_from_file_location(
                    dox_name, dox_fn)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception as exc:
            raise DOXError(
                "Error loading {}!".format(dox_fn)) from exc

        print(self.ctx.registered_experiments)

    def run(self):
        """
        Run the experiments of this DOX.
        """

        print("DOX.run")

        self.ctx.run()
