import pytest

from experitur.recursive_formatter import RecursiveFormatter, RecursiveDict


def test_RecursiveFormatter():
    assert RecursiveFormatter().format("{}", "foo") == "foo"
    assert RecursiveFormatter().format("{bar}", bar="foo") == "foo"

    assert RecursiveFormatter().format("{0}", 1) == 1
    assert RecursiveFormatter().format("{bar}", bar=1) == 1

    assert RecursiveFormatter().format("{!s}", 1) == "1"
    assert RecursiveFormatter().format("{:.2f}", 1) == "1.00"

    with pytest.raises(KeyError):
        assert RecursiveFormatter().format("{missing}")

    assert RecursiveFormatter(allow_missing=True).format(
        "{missing}") == "{missing}"

    assert RecursiveFormatter().format(
        "{foo_{bar}}", bar="baz", foo_baz="foo") == "foo"


def test_RecursiveDict():
    m = {"a": "{a-{b}}-{f}",
         "b": "{c}",
         "a-foo": "foo",
         "a-bar": "bar",
         "c": "foo",
         "d": 1,
         "e": "{d}"
         }

    mr = RecursiveDict(m, allow_missing=True)

    assert mr["a"] == "foo-{f}"
    assert mr["b"] == "foo"
    assert mr["c"] == "foo"
    assert mr["d"] == 1
    assert mr["e"] == 1

    assert list(iter(mr)) == list(iter(m))
    assert len(mr) == len(m)
    assert mr.as_dict() == {'a': 'foo-{f}', "b": "foo", 'a-bar': 'bar', 'a-foo': 'foo',
                            "c": "foo", "d": 1, "e": 1}
