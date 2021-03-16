import collections.abc
import itertools
from typing import Mapping, Iterable, Tuple, Any, Union


def merge_dicts(
    a: Mapping, b: Union[Mapping, Iterable[Tuple[Any, Any]], None] = None, **kwargs
):
    """
    Recursively merge b into a copy of a, overwriting existing values.

    Parameters
    ----------
    a
    """

    # Copy a
    a = {**a}

    itemiters = []

    if isinstance(b, collections.abc.Mapping):
        itemiters.append(b.items())
    elif isinstance(b, collections.abc.Iterable):
        itemiters.append(b)  # type: ignore
    elif b is None:
        pass
    else:
        raise ValueError(f"Unexpected type for b: {type(b)}")

    itemiters.append(kwargs.items())

    for key, value in itertools.chain(*itemiters):
        if (
            key in a
            and isinstance(a[key], collections.abc.MutableMapping)
            and isinstance(value, collections.abc.Mapping)
        ):
            a[key] = merge_dicts(a[key], value)
        else:
            a[key] = value
    return a
