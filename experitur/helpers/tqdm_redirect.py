import contextlib
import sys
import threading

from tqdm import tqdm


class _TQDMFile(object):
    def __init__(self, file):
        self.file = file

        # We need the Python implementation here to get access to _count
        self.lock = threading._RLock()

    def write(self, x):
        with self.lock:
            # If recursion occured, write directly to file
            if self.lock._count > 1:
                self.file.write(x)
            else:
                # Otherwise write through tqdm to not mess with the progress bars
                tqdm.write(x, end="", file=self.file)

    def __eq__(self, other):
        return other is self.file

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_stdout():
    """Redirect sys.stdout through tqdm to avoid the breaking of progress bars."""
    save_stdout = sys.stdout
    sys.stdout = _TQDMFile(sys.stdout)
    yield
    sys.stdout = save_stdout
