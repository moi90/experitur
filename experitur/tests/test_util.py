from experitur import util
import pytest
import random
import string

try:
    choices = random.choices
except AttributeError:
    def choices(seq, k=1):
        return [random.choice(seq) for i in range(k)]


@pytest.fixture(name="prefixes")
def fixture_prefixes():
    return [
        ''.join(choices(string.ascii_uppercase + string.digits, k=N)) + '_'
        for N in range(1, 4)
    ]


def test_extract_parameters(prefixes):
    seed = {"a": 1, "b": 2, "c": 3}
    parameters = {p+k: v for p in prefixes for k, v in seed.items()}

    for p in prefixes:
        assert util.extract_parameters(p, parameters) == seed


def noop(x, y, a, b, foo=None, bar=99):
    return (x, y, a, b, foo, bar)


def test_apply_parameters(recwarn):
    parameters = {
        "prefix_a": 1,
        "prefix_b": 2,
        "prefix_c": 3,  # c is not a parameter of noop
        "prefix2_a": 4  # prefix2 is not used
    }
    result = util.apply_parameters("prefix_", parameters, noop, 5, 6, foo=3)
    assert result == (5, 6, 1, 2, 3, 99)

    result = util.apply_parameters(
        "prefix_", parameters, noop, 5, 6, foo=3, a=10)
    assert result == (5, 6, 10, 2, 3, 99)
    assert recwarn.pop(UserWarning)


def test_set_default_parameters():
    parameters = {
        "p_foo": "foo"
    }

    util.set_default_parameters("p_", parameters, noop, x=12)

    assert parameters == {
        "p_x": 12,
        "p_foo": "foo",
        "p_bar": 99,
    }

    with pytest.raises(ValueError):
        util.set_default_parameters("p_", parameters, noop, 10, x=12)
