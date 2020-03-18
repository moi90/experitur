from experitur.helpers.merge_dicts import merge_dicts
import pytest


def test_merge_dicts():
    a = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4, "f": {"g": 5, "h": 6}}

    b = {"b": {"c": 7}, "f": 8}

    result = merge_dicts(a, b)

    assert result == {"a": 1, "b": {"c": 7, "d": 3}, "e": 4, "f": 8}

    assert result is not a

    merge_dicts(a, b=None)

    result = merge_dicts(a, list(b.items()), f=10)
    assert result == {"a": 1, "b": {"c": 7, "d": 3}, "e": 4, "f": 10}

    with pytest.raises(ValueError):
        merge_dicts(a, 1)
