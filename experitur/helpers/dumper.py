import yaml


class ExperiturDumper(yaml.Dumper):
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
    ExperiturDumper.add_representer(np.ndarray, ExperiturDumper.ndarray_representer)
    ExperiturDumper.add_multi_representer(np.number, ExperiturDumper.number_representer)
