import contextlib
import sys

from tqdm import tqdm


class TqdmFile(object):
    def __init__(self, file):
        self.file = file

    def write(self, x):
        tqdm.write(x, end="", file=self.file)

    def __eq__(self, other):
        return other is self.file

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_stdout():
    save_stdout = sys.stdout
    sys.stdout = TqdmFile(sys.stdout)
    yield
    sys.stdout = save_stdout
