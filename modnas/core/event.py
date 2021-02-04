from functools import wraps
from .singleton import singleton


@singleton
class EventManager():

    def __init__(self):
        self.handlers = {}
        self.event_queue = []

    def get_handlers(self, ev):
        ev_handlers = self.handlers.get(ev, [])
        for p, h in ev_handlers:
            yield h

    def on(self, ev, handler, priority=0):
        ev_handlers = self.handlers.get(ev, [])
        ev_handlers.append((priority, handler))
        ev_handlers.sort(key=lambda s: -s[0])
        self.handlers[ev] = ev_handlers

    def fire(self, ev, *args, callback=None, delayed=False, **kwargs):
        self.event_queue.append((ev, args, kwargs, callback))
        if delayed:
            return
        return self.dispatch_all()[ev]

    def off(self, ev, handler):
        ev_handlers = self.handlers.get(ev, None)
        if ev_handlers is None:
            return
        for i, (p, h) in enumerate(ev_handlers):
            if h == handler:
                ev_handlers.pop(i)
                break
        if not ev_handlers:
            del self.handlers[ev]

    def dispatch_all(self):
        rets = {}
        for ev_spec in self.event_queue:
            ev, args, kwargs, callback = ev_spec
            ret = None
            for handler in self.get_handlers(ev):
                ret = handler(*args, **kwargs)
            if callback is not None:
                callback(ret)
            rets[ev] = ret
        self.event_queue.clear()
        return rets


def event_hooked(func, name=None, before=True, after=True, qual=True):
    name = name or (func.__qualname__ if qual else func.__name__)

    @wraps(func)
    def wrapped(*args, **kwargs):
        if wrapped.before:
            hret = EventManager().fire('before:' + name, *args, **kwargs)
            if hret is not None:
                args, kwargs = hret[0] or args, hret[1] or kwargs
        fret = func(*args, **kwargs)
        if wrapped.after:
            hret = EventManager().fire('after:' + name, fret, *args, **kwargs)
            if hret is not None:
                fret = hret
        return fret
    wrapped._event_unhooked = func
    wrapped.before = before
    wrapped.after = after
    return wrapped


def event_unhooked(func, remove_all=False, before=False, after=False):
    func.before = before
    func.after = after
    if remove_all:
        return func._event_unhooked
    return func


event_on = EventManager().on
event_off = EventManager().off
