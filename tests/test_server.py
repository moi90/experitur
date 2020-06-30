from experitur.server import ExperiturServer
from experitur.core.context import Context

try:
    import gevent
    import zerorpc
except ImportError:
    pass
else:

    def test_server(tmp_path, random_ipc_endpoint):
        config = {"remote_endpoint": random_ipc_endpoint}
        with Context(str(tmp_path), config) as ctx:
            server = ExperiturServer(ctx)
            server.bind(random_ipc_endpoint)
            gevent.spawn(server.run)

            client = zerorpc.Client(random_ipc_endpoint)

            client.set_trial_data("some_id", {"wdir": ".", "id": "foo"})

    def test_zerorpc(random_ipc_endpoint):
        class MySrv(zerorpc.Server):
            def __init__(self):
                super().__init__()
                self.ctx = None

            def lock(self):
                pass

            def release(self):
                pass

            def get_trial_data(self, trial_id):
                pass

            def set_trial_data(self, trial_id, trial_data):
                pass

        srv = MySrv()
        srv.bind(random_ipc_endpoint)
        gevent.spawn(srv.run)

        client = zerorpc.Client()
        client.connect(random_ipc_endpoint)

        client.set_trial_data("trial_id", {"wdir": ".", "id": "foo"})
