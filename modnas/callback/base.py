from modnas.core.event import event_on, event_off


class CallbackBase():

    def __init__(self, handlers=None) -> None:
        self.handlers = handlers
        self.bind_handlers()

    def bind_handlers(self):
        for ev, h in self.handlers.items():
            event_on(ev, h, self.priority)

    def unbind_handlers(self):
        for ev, h in self.handlers.items():
            event_off(ev, h)
