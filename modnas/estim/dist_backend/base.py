import threading


class RemoteBase():
    def __init__(self):
        super().__init__()
        self.on_done = None
        self.on_failed = None

    def call(self, func, *args, on_done=None, on_failed=None, **kwargs):
        self.on_done = on_done
        self.on_failed = on_failed
        self.th_rpc = threading.Thread(target=self.rpc, args=(func,) + args, kwargs=kwargs)
        self.th_rpc.start()

    def rpc(self, func, *args, **kwargs):
        raise NotImplementedError

    def on_rpc_done(self, ret):
        self.ret = ret
        self.on_done(ret)

    def on_rpc_failed(self, ret):
        self.on_failed(ret)


class WorkerBase():

    def run(self, estim):
        raise NotImplementedError
