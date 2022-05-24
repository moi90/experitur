from yaml import dump, load, load_all

try:
    from yaml import CDumper as _Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper as _Dumper
    from yaml import Loader as Loader


class Dumper(_Dumper):
    def ndarray_representer(self, data):
        return self.represent_list(data.tolist())

    def number_representer(self, data):
        return self.represent_data(data.tolist())


try:
    import numpy as np
except ImportError:  # pragma: no cover
    pass
else:
    # If numpy is available, define YAML Emitter, Serializer, Representer, Resolver
    Dumper.add_representer(np.ndarray, Dumper.ndarray_representer)
    Dumper.add_multi_representer(np.number, Dumper.number_representer)
