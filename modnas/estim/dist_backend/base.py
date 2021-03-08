"""Distributed remote client and server."""
import threading


class RemoteBase():
    """Distributed remote client class."""

    def __init__(self):
        super().__init__()
        self.on_done = None
        self.on_failed = None

    def call(self, func, *args, on_done=None, on_failed=None, **kwargs):
        """Call function on remote client with callbacks."""
        self.on_done = on_done
        self.on_failed = on_failed
        self.th_rpc = threading.Thread(target=self.rpc, args=(func,) + args, kwargs=kwargs)
        self.th_rpc.start()

    def rpc(self, func, *args, **kwargs):
        """Call function on remote client."""
        raise NotImplementedError

    def on_rpc_done(self, ret):
        """Invoke callback when remote call finishes."""
        self.ret = ret
        self.on_done(ret)

    def on_rpc_failed(self, ret):
        """Invoke callback when remote call fails."""
        self.on_failed(ret)


class WorkerBase():
    """Distributed remote worker (server) class."""

    def run(self, estim):
        """Run worker."""
        raise NotImplementedError
