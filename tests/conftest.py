import pytest


@pytest.fixture()
def random_ipc_endpoint(tmp_path):
    return f"ipc://{tmp_path!s}/ipc"
