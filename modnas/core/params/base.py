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
        if on_update is not None:
            event_on('update:' + self.name, on_update)

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
        event_emit('update:' + self.name, self)

    def __deepcopy__(self, memo):
        # disable deepcopy
        return self
