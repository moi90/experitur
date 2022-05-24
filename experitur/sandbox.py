import os
import threading
from collections import deque
from typing import Deque, Dict, List, Optional, TYPE_CHECKING
import shlex
import subprocess

if TYPE_CHECKING:
    from experitur.core.trial import Trial

from multiprocessing.connection import Connection, Pipe


class CommunicationError(Exception):
    pass


class Worker:
    def __init__(self, parent: "WorkerPool") -> None:
        self.parent = parent

        self._child: Optional[subprocess.Popen] = None
        self._connection: Optional[Connection] = None
        self._error: Optional[Exception] = None
        self._child_ready = threading.Condition()

        initializer_thread = threading.Thread(target=self._initialize_child)
        initializer_thread.start()

    def _initialize_child(self):
        parent_conn, child_conn = Pipe(duplex=True)

        # Child subprocess
        cmd = shlex.split(self.parent.child_command) + [
            "private-run-trial",
            self.parent.dox_fn,
            str(child_conn.fileno()),
        ]

        print(f"Starting subprocess `{' '.join(cmd)}`...")
        child = subprocess.Popen(cmd, shell=False, pass_fds=[child_conn.fileno()])
        child_conn.close()

        try:
            # Wait for "ready" message
            msg = parent_conn.recv()

            if msg != ("ready",):
                raise CommunicationError(f"Unexpected message: {msg}")

            with self._child_ready:
                self._child = child
                self._connection = parent_conn
                self._child_ready.notify_all()
        except Exception as exc:
            parent_conn.close()
            child.terminate()
            child.wait()
            self._error = exc
            print(
                f"Exception durin worker initialization: {exc}. Returncode: {child.returncode}"
            )
        finally:
            self.parent.notify_worker(self)

    @property
    def terminated(self):
        with self._child_ready:
            if self._child is None:
                return False

            return self._child.returncode is not None

    @property
    def error(self):
        with self._child_ready:
            return self._error

    def run_trial(self, trial_data: Dict) -> Dict:
        # Wait until child is ready
        with self._child_ready:
            self._child_ready.wait_for(lambda: self._child is not None)

        # Send a job
        self._connection.send(("run", trial_data))

        # Receive response
        status, msg = self._connection.recv()

        if status == "error":
            exc_type, exc_msg = msg
            raise exc_type(exc_msg)

        if status == "success" and isinstance(msg, Dict):
            return msg

        raise CommunicationError(f"Unexpected response: {(status, msg)}")


class WorkerPool:
    def __init__(self, dox_fn, n_workers: int = 1, child_command: Optional[str] = None):
        self.dox_fn = dox_fn
        self.n_workers = n_workers
        self.child_command = child_command if child_command is not None else "experitur"

        self._workers: List[Worker] = []
        self._ready_queue: Deque[Worker] = deque()
        self._ready = threading.Condition()

        self._setup()

    def _setup(self):
        self._workers = [w for w in self._workers if not w.terminated]

        for _ in range(self.n_workers - len(self._workers)):
            self._workers.append(Worker(self))

    def notify_worker(self, worker: Worker):
        with self._ready:
            self._ready_queue.append(worker)
            self._ready.notify_all()

    def _get_ready_worker(self):
        with self._ready:
            self._ready.wait_for(lambda: len(self._ready_queue) > 0)
            return self._ready_queue.popleft()

    def run_trial(self, trial_data: Dict) -> Dict:
        worker = self._get_ready_worker()

        if worker.error is not None:
            raise worker.error

        try:
            return worker.run_trial(trial_data)
        finally:
            self._setup()