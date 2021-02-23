import rpyc
from rpyc.utils.server import ThreadedServer
from .base import RemoteBase
from modnas.registry.dist_remote import register as register_remote
from modnas.registry.dist_worker import register as register_worker


@register_remote
class RPyCRemote(RemoteBase):
    def __init__(self, address, port=18861):
        super().__init__()
        self.conn = rpyc.connect(address, port)

    def rpc(self, func, *args, **kwargs):
        ret = self.conn.root.estim_call(func, *args, **kwargs)
        self.on_rpc_done(ret)


def convert_normal(obj):
    if isinstance(obj, dict):
        return {k: obj[k] for k in obj}
    return obj


class ModNASService(rpyc.Service):

    def exposed_get_estim(self):
        return self.estim

    def exposed_estim_call(self, func, *args, **kwargs):
        args = [convert_normal(a) for a in args]
        kwargs = {k: convert_normal(v) for k, v in kwargs.items()}
        return getattr(self.estim, func)(*args, **kwargs)


@register_worker
class RPyCWorker():
    def __init__(self, *args, port=18861, **kwargs):
        self.server = ThreadedServer(ModNASService, *args, port=port, **kwargs)

    def run(self, estim):
        self.server.service.estim = estim
        self.server.start()
