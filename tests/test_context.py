import pytest

from experitur.core.context import (
    push_context,
    default_context,
    Context,
    DependencyError,
)


def test_push_context():
    """
    Make sure that the push_context context manager works in the expected way.
    """
    orig_ctx = default_context

    with push_context() as new_ctx:
        assert new_ctx != orig_ctx

    assert default_context == orig_ctx


def test__order_experiments_fail(tmp_path):
    with push_context(Context(str(tmp_path))) as ctx:
        # Create a dependency circle
        a = ctx.experiment("a")
        b = ctx.experiment("b", parent=a)
        a.parent = b

        with pytest.raises(DependencyError):
            ctx.run()


def test_dependencies(tmp_path):
    with push_context(Context(str(tmp_path))) as ctx:

        @ctx.experiment("a")
        def a(trial):
            pass

        b = ctx.experiment("b", parent=a)

        ctx.run([b])


def test_merge_config(tmp_path):
    config = {
        k: not v for k, v in Context._default_config.items() if isinstance(v, bool)
    }

    config["a"] = 1
    config["b"] = 2
    config["c"] = 3

    with push_context(Context(str(tmp_path), config=config.copy())) as ctx:
        assert all(v == ctx.config[k] for k, v in config.items())
