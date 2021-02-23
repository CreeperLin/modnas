import threading
from modnas.registry.dist_remote import register as register_remote, build
from .base import RemoteBase


@register_remote
class ManagedRemotes(RemoteBase):
    def __init__(self, remote_conf):
        super().__init__()
        remotes = {}
        for k, conf in remote_conf.items():
            remotes[k] = build(conf)
        self.remotes = remotes
        self.idle = {k: True for k in remote_conf.keys()}
        self.idle_cond = threading.Lock()

    def add_remote(self, key, rmt):
        self.remotes[key] = rmt
        self.idle[key] = True

    def del_remote(self, key):
        del self.remotes[key]
        del self.idle[key]

    def is_idle(self, key):
        return self.idle[key]

    def idle_remotes(self):
        return [k for k, v in self.idle.items() if v]

    def get_idle_remote(self, busy=True, wait=True):
        idle_rmt = None
        while idle_rmt is None:
            idles = self.idle_remotes()
            if not idles:
                if not wait:
                    return None
                self.idle_cond.acquire()
                self.idle_cond.release()
            else:
                idle_rmt = idles[0]
        if busy:
            self.set_idle(idle_rmt, False)
        return idle_rmt

    def set_idle(self, key, idle=True):
        self.idle[key] = idle
        if idle and self.idle_cond.locked():
            self.idle_cond.release()
        elif not self.idle_remotes():
            self.idle_cond.acquire()

    def call(self, *args, on_done=None, on_failed=None, **kwargs):
        def wrap_cb(cb, r):
            def wrapped(*args, **kwargs):
                self.set_idle(r)
                return None if cb is None else cb(*args, **kwargs)
            return wrapped

        rmt_key = self.get_idle_remote()
        if rmt_key is None:
            return
        on_done = wrap_cb(on_done, rmt_key)
        on_failed = wrap_cb(on_failed, rmt_key)
        self.remotes[rmt_key].call(*args, on_done=on_done, on_failed=on_failed, **kwargs)
