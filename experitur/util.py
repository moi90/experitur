from collections.abc import Iterable, Mapping
from typing import Dict


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
    if isinstance(obj, (str, bytes)):
        return [obj]

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


def format_parameters(parameters: Mapping):
    return ", ".join(f"{k}={v}" for k, v in sorted(parameters.items()))


def freeze(value):
    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, set):
        return frozenset(value)

    return value


def isatty(stream: IO):
    try:
        return stream.isatty()
    except AttributeError:
        return False


def cprint(*args, color=None, on_color=None, attrs=None, file=sys.stdout, **kwargs):
    """
    Print styled text if connected to a tty.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.
    """
    if isatty(file):
        text = " ".join(str(o) for o in args)
        termcolor.cprint(
            text, color=color, on_color=on_color, attrs=attrs, file=file, **kwargs
        )
    else:
        print(*args, file=file, **kwargs)

class _Unset:
    """
    Singleton to signify an unset value.

    Warning: If a value is not set, the trial might be skipped if a trial with this value set exists!
    
    """

    def __repr__(self):
        return "<unset>"


unset = _Unset()
del _Unset


def clean_unset(mapping: Mapping) -> Dict:
    return {k: v for k, v in mapping.items() if v is not unset}

