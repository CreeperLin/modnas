from modnas.core.event import event_on, event_off
from modnas.utils.logging import get_logger


class CallbackBase():

    logger = get_logger('callback')
    priority = 0

    def __init__(self, handler_conf=None) -> None:
        self.handlers = None
        self.bind_handlers(handler_conf)

    def bind_handlers(self, handler_conf):
        handlers = {}
        for ev, conf in handler_conf.items():
            prio = None
            if isinstance(conf, (list, tuple)):
                h = conf[0]
                if len(conf) > 1:
                    prio = conf[1]
            else:
                h = conf
            event_on(ev, h, self.priority if prio is None else prio)
            handlers[ev] = h
        self.handlers = handlers

    def unbind_handlers(self):
        for ev, h in self.handlers.items():
            event_off(ev, h)
