from collections import OrderedDict
from ..event import event_emit, event_on
from ..param_space import ParamSpace


class Param():
    def __init__(self, name=None, space=None, on_update=None):
        self.name = None
        self._parent = None
        self._children = OrderedDict()
        space = space or ParamSpace()
        space.register(self, name)
        self.event_name = 'update:{}'.format(self.name)
        if on_update is not None:
            event_on(self.event_name, on_update)
        set_value_ori = self.set_value

        def set_value_hooked(*args, **kwargs):
            set_value_ori(*args, **kwargs)
            self.on_update()
        self.set_value = set_value_hooked

    def __repr__(self):
        return '{}(name={}, {})'.format(self.__class__.__name__, self.name, self.extra_repr())

    def extra_repr(self):
        return ''

    def is_valid(self, value):
        return True

    def value(self):
        return self.val

    def set_value(self, value):
        if not self.is_valid(value):
            raise ValueError('Invalid parameter value')
        self.val = value

    def on_update(self):
        event_emit(self.event_name, self)

    def __deepcopy__(self, memo):
        # disable deepcopy
        return self
