from inspect import signature
import warnings


def extract_parameters(prefix, parameters):
    """
    Extract parameters beginning with prefix and remove the prefix.
    """
    start = len(prefix)

    return {
        k[start:]: v
        for k, v in parameters.items()
        if k.startswith(prefix)
    }


def apply_parameters(prefix, parameters, callable_, *args, **kwargs):
    callable_names = set(
        param.name
        for param in signature(callable_).parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD)

    start = len(prefix)
    parameters = {
        k[start:]: v
        for k, v in parameters.items()
        if k.startswith(prefix) and k[start:] in callable_names
    }

    intersection_names = set(kwargs.keys()) & set(parameters.keys())
    if intersection_names:
        warnings.warn("Redefining parameter(s) {} with keyword parameter.".format(
            ", ".join(intersection_names)))

    for k, v in kwargs.items():
        parameters[k] = v

    return callable_(*args, **parameters)


def set_default_parameters(prefix, parameters, *args, **defaults):
    """
    Set default parameters.

    Default parameters can be assigned directly or guessed from a callable.
    """
    if len(args) > 1:
        raise ValueError("Only 2 or 3 positional arguments allowed.")

    # First set explicit defaults
    for name, value in defaults.items():
        parameters.setdefault(prefix + name, value)

    if args and callable(args[0]):
        callable_ = args[0]
        for param in signature(callable_).parameters.values():
            if param.default is not param.empty:
                parameters.setdefault(prefix + param.name, param.default)
