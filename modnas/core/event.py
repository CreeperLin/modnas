import inspect
from functools import wraps
from . import singleton, make_decorator


@singleton
class EventManager():

    def __init__(self):
        self.handlers = {}
        self.event_queue = []

    def reset(self):
        self.handlers.clear()
        self.event_queue.clear()

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
        if ev not in self.handlers:
            return
        self.event_queue.append((ev, args, kwargs, callback))
        if delayed:
            return
        return self.dispatch_all()[ev]

    def off(self, ev, handler=None):
        ev_handlers = self.handlers.get(ev, None)
        if ev_handlers is None:
            return
        if handler is None:
            del self.handlers[ev]
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


@make_decorator
def event_hooked(func, name=None, before=True, after=True, qual=True, module=False):
    qual = func.__qualname__.split('.')[0] if qual is True else (None if qual is False else qual)
    module = func.__module__ if module is True else (None if module is False else module)
    name = name or func.__name__
    ev = (module + '.' if module else '') + (qual + '.' if qual else '') + name

    @wraps(func)
    def wrapped(*args, **kwargs):
        if wrapped.before:
            hret = EventManager().fire('before:' + ev, *args, **kwargs)
            if hret is not None:
                args, kwargs = hret[0] or args, hret[1] or kwargs
        fret = func(*args, **kwargs)
        if wrapped.after:
            hret = EventManager().fire('after:' + ev, fret, *args, **kwargs)
            if hret is not None:
                fret = hret
        return fret
    wrapped._event_unhooked = func
    wrapped.before = before
    wrapped.after = after
    return wrapped


@make_decorator
def event_unhooked(func, remove_all=False, before=False, after=False):
    func.before = before
    func.after = after
    if remove_all:
        return func._event_unhooked
    return func


@make_decorator
def event_hooked_members(obj, *args, methods=None, is_method=False, is_function=False, **kwargs):
    for name, attr in inspect.getmembers(obj):
        if methods is not None and name not in methods:
            continue
        if is_method and not inspect.ismethod(attr):
            continue
        if is_function and not inspect.isfunction(attr):
            continue
        setattr(obj, name, event_hooked(attr, *args, **kwargs))
    return obj


@make_decorator
def event_hooked_inst(cls, *args, **kwargs):
    @wraps(cls)
    def wrapped(*cls_args, **cls_kwargs):
        inst = cls(*cls_args, **cls_kwargs)
        event_hooked_members(inst, *args, is_method=True, **kwargs)
        return inst
    return wrapped


@make_decorator
def event_hooked_class(cls, *args, **kwargs):
    event_hooked_members(cls, *args, is_function=True, **kwargs)
    return cls


@make_decorator
def event_hooked_subclass(cls, base_qual=True, *args, **kwargs):
    ori_init = cls.__init__

    def new_init(self, *fn_args, **fn_kwargs):
        ori_init(self, *fn_args, **fn_kwargs)
        if base_qual:
            kwargs['qual'] = cls.__name__
        event_hooked_members(self, *args, is_method=True, **kwargs)
    setattr(cls, '__init__', new_init)
    return cls


event_on = EventManager().on
event_off = EventManager().off
event_fire = EventManager().fire
