from collections.abc import Iterable, Mapping


def callable_to_name(obj):
    if callable(obj):
        return "{}.{}".format(obj.__module__, obj.__name__)

    if isinstance(obj, list):
        return [callable_to_name(x) for x in obj]

    if isinstance(obj, dict):
        return {callable_to_name(k): callable_to_name(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(callable_to_name(x) for x in obj)

    return obj


def ensure_list(obj):
    """Accepts a thing, a list of things or None and turns it into a list."""
    if isinstance(obj, Iterable):
        return list(obj)

    if obj is None:
        return []

    return [obj]


def ensure_dict(obj):
    """Accepts a dict or None and turns it into a dict."""
    if isinstance(obj, Mapping):
        return dict(obj)

    if obj is None:
        return {}

    raise ValueError("Expected mapping or None, got {obj!r}")
