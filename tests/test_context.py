from experitur import context


def test_push_context():
    """
    Make sure that the push_context context manager works in the expected way.
    """
    orig_ctx = context.default_context

    with context.push_context() as new_ctx:
        assert new_ctx != orig_ctx

    assert context.default_context == orig_ctx
