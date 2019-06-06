import collections.abc
import os.path
import glob
import yaml
import shutil
import yaml.dumper


class YAMLDict(collections.abc.MutableMapping):
    """
    A dictionary-like object that stores the data behind each key in a yaml file.
    """

    def __init__(self, root, pattern=None, dumper=yaml.dumper.Dumper):
        self.root = root
        self.pattern = pattern
        self.dumper = dumper

        if self.pattern is None:
            self.pattern = "{}.yaml"

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, key):
        path = os.path.join(self.root, self.pattern.format(key))

        with open(path) as fp:
            return yaml.load(fp)

    def __iter__(self):
        path = os.path.join(self.root, self.pattern.format("*"))

        left, right = path.split("*", 1)

        for entry_fn in glob.iglob(path):
            if os.path.isdir(entry_fn):
                continue

            # Convert entry_fn back to key
            k = entry_fn[len(left):-len(right)]

            yield k

    def __setitem__(self, key, value):
        path = os.path.join(self.root, self.pattern.format(key))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as fp:
            yaml.dump(value, fp, Dumper=self.dumper)

        #raise KeyError

    def __delitem__(self, key):
        path = os.path.join(self.root, self.pattern.format(key))

        try:
            os.remove(path)
        except FileNotFoundError:
            raise KeyError

        shutil.rmtree(os.path.dirname(path))
