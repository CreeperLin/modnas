import random
import numpy as np
from .base import Param
from ...registry.params import register


def default_categorical_sampler(dim):
    return np.random.randint(dim)


def default_int_sampler(bound):
    return random.randint(*bound)


def default_real_sampler(bound):
    return random.uniform(*bound)


@register
class Categorical(Param):
    TYPE = 'C'

    def __init__(self, choices, sampler=None, name=None, space=None, on_update=None):
        super().__init__(name, space, on_update)
        self.sample = default_categorical_sampler if sampler is None else sampler
        self.choices = choices
        self._length = None
        self.val = None

    def extra_repr(self):
        return 'choices={}'.format(self.choices)

    def is_valid(self, value):
        return value in self.choices

    def get_value(self, index):
        return self.choices[index]

    def set_value(self, value):
        index = self.get_index(value)
        self.val = index

    def value(self):
        return self.choices[self.index()]

    def index(self):
        if self.val is None:
            self.val = self.sample(len(self.choices))
        return self.val

    def get_index(self, value):
        return self.choices.index(value)

    def set_index(self, index):
        self.val = index

    def __len__(self):
        if self._length is None:
            self._length = len(self.choices)
        return self._length


@register
class Numeric(Param):
    TYPE = 'N'

    def __init__(self, low, high, ntype=None, sampler=None, name=None, space=None, on_update=None):
        super().__init__(name, space, on_update)
        self.bound = (low, high)
        self.ntype = 'i' if (all(isinstance(b, int) for b in self.bound) and ntype != 'r') else 'r'
        default_sampler = default_int_sampler if self.ntype == 'i' else default_real_sampler
        self.sample = default_sampler if sampler is None else sampler
        self.val = None

    def extra_repr(self):
        return 'type={}, range={}'.format(self.ntype, self.bound)

    def is_valid(self, value):
        return self.bound[0] <= value <= self.bound[1]

    def set_value(self, value):
        if not self.is_valid(value):
            raise ValueError('invalid numeric parameter value')
        self.val = value

    def value(self):
        if self.val is None:
            self.val = self.sample(self.bound)
        return self.val

    def is_int(self):
        return self.ntype == 'i'

    def is_real(self):
        return self.ntype == 'r'
