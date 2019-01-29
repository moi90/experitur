import yaml
import numpy as np


class ExperiturDumper(yaml.Dumper):
    def ndarray_representer(self, data):
        return self.represent_dict(data.tolist())

    def number_representer(self, data):
        return self.represent_data(data.tolist())


#Emitter, Serializer, Representer, Resolver
ExperiturDumper.add_representer(
    np.ndarray, ExperiturDumper.ndarray_representer)
ExperiturDumper.add_multi_representer(
    np.number, ExperiturDumper.number_representer)
