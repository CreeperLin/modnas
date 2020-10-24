"""Architecture Parameter Space."""
import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import random


class ParamSpace():
    """Parameter Space class."""

    def __init__(self):
        self.reset()

    def reset(self, ):
        """Reset parameter space."""
        self._param_id = -1
        self._params_map = OrderedDict()
        self._categorical_length = None

    def register(self, param, name):
        """Register a new parameter."""
        param.pid = self.new_param_id()
        if name is None:
            reg_name = self.new_param_name(param)
        else:
            reg_name = name
            idx = 0
            while reg_name in self._params_map:
                idx += 1
                reg_name = '{}_{}'.format(name, idx)
        param.name = reg_name
        self.add_param(reg_name, param)
        if isinstance(param, ParamCategorical):
            self._categorical_length = None

    def new_param_id(self, ):
        """Return a new parameter id."""
        self._param_id += 1
        return self._param_id

    def new_param_name(self, param):
        """Return a new parameter name."""
        if isinstance(param, ParamTensor):
            prefix = 't'
        elif isinstance(param, ParamCategorical):
            prefix = 'c'
        elif isinstance(param, ParamNumeric):
            prefix = 'n'
        else:
            prefix = 'm'
        return '{}_{}'.format(prefix, self._param_id)

    def params(self, ):
        """Return an iterator over parameters."""
        for p in self._params_map.values():
            yield p

    def named_params(self, ):
        """Return an iterator over named parameters."""
        for n, p in self._params_map.items():
            yield n, p

    def add_param(self, name, param):
        """Add a parameter to space."""
        self._params_map[name] = param

    def get_param(self, name):
        """Return a parameter by name."""
        return self._params_map.get(name, None)

    def categorical_size(self, ):
        """Return size of the categorical parameter space."""
        if self._categorical_length is None:
            prod = 1
            for x in self.categorical_params():
                prod *= len(x)
            self._categorical_length = prod
        return self._categorical_length

    def categorical_params(self, ):
        """Return an iterator over categorical parameters."""
        for p in self.params():
            if isinstance(p, ParamCategorical):
                yield p

    def tensor_params(self, ):
        """Return an iterator over tensor parameters."""
        for p in self.params():
            if isinstance(p, ParamTensor):
                yield p

    def categorical_values(self, ):
        """Return an iterator over categorical parameters values."""
        for p in self.params():
            if isinstance(p, ParamCategorical):
                yield p.value()

    def tensor_values(self, ):
        """Return an iterator over tensor parameters values."""
        for p in self.params():
            if isinstance(p, ParamTensor):
                yield p.value()

    def get_categorical_params(self, idx):
        """Return a set of parameter values from a categorical space index."""
        arch_param = OrderedDict()
        for ap in self.categorical_params():
            ap_dim = len(ap)
            arch_param[ap.name] = ap.get_value(idx % ap_dim)
            idx //= ap_dim
        return arch_param

    def get_categorical_index(self, param):
        """Return a categorical space index from a set of parameter values."""
        idx = 0
        base = 1
        for p in self.categorical_params():
            p_dim = len(p)
            p_idx = p.get_index(param[p.name])
            idx += base * p_idx
            base *= p_dim
        return idx

    def update_params(self, pmap, trigger=True):
        """Update parameter values from a dict."""
        for k, v in pmap.items():
            p = self.get_param(k)
            if p is None:
                logging.error('parameter \'{}\' not found'.format(k))
            p.set_value(v)
            if trigger:
                p.on_update()

    def on_update_tensor_params(self):
        """Invoke handlers on tensor parameter updates."""
        for ap in self.tensor_params():
            ap.on_update()


class Param():
    """Parameter class."""

    def __init__(self, space, name, on_update):
        self.name = None
        self.on_update_handler = on_update
        space = space or ArchParamSpace
        space.register(self, name)

    def __repr__(self):
        """Return string representation of param."""
        return '{}(name={}, {})'.format(self.__class__.__name__, self.name, self.extra_repr())

    def extra_repr(self):
        """Return extra string representation of param."""
        return ''

    def is_valid(self, value):
        """Return True if a value is valid for this parameter."""
        return True

    def set_value(self, value):
        """Set parameter value."""
        raise NotImplementedError

    def value(self):
        """Return parameter value."""
        raise NotImplementedError

    def on_update(self):
        """Call update handler on this parameter."""
        handler = self.on_update_handler
        if handler is not None:
            return handler(self)
        return None

    def __deepcopy__(self, memo):
        """Disable deepcopy for parameter."""
        # disable deepcopy
        return self


def _default_categorical_sampler(dim):
    return np.random.randint(dim)


def _default_tensor_sampler(shape):
    _init_ratio = 1e-3
    return nn.Parameter(_init_ratio * torch.randn(shape))


def _default_int_sampler(bound):
    return random.randint(*bound)


def _default_real_sampler(bound):
    return random.uniform(*bound)


class ParamNumeric(Param):
    """Numerical Parameter class."""

    def __init__(self, low, high, space=None, ntype=None, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
        self.bound = (low, high)
        self.ntype = 'i' if (all(isinstance(b, int) for b in self.bound) and ntype != 'r') else 'r'
        default_sampler = _default_int_sampler if self.ntype == 'i' else _default_real_sampler
        self.sample = default_sampler if sampler is None else sampler
        self.val = None

    def extra_repr(self):
        """Return extra string representation of param."""
        return 'type={}, range={}'.format(self.ntype, self.bound)

    def is_valid(self, value):
        """Return True if a value is valid for this parameter."""
        return self.bound[0] <= value <= self.bound[1]

    def set_value(self, value):
        """Set parameter value."""
        if not self.is_valid(value):
            raise ValueError('invalid numeric parameter value')
        self.val = value

    def value(self):
        """Return parameter value."""
        if self.val is None:
            self.val = self.sample(self.bound)
        return self.val

    def is_int(self):
        """Return True if parameter is int type."""
        return self.ntype == 'i'

    def is_real(self):
        """Return True if parameter is float type."""
        return self.ntype == 'r'


class ParamCategorical(Param):
    """Categorical Parameter class."""

    def __init__(self, choices, space=None, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
        self.sample = _default_categorical_sampler if sampler is None else sampler
        self.choices = choices
        self._length = None
        self.val = None

    def extra_repr(self):
        """Return extra string representation of param."""
        return 'choices={}'.format(self.choices)

    def is_valid(self, value):
        """Return True if a value is valid for this parameter."""
        return value in self.choices

    def get_value(self, index):
        """Return parameter value of an index."""
        return self.choices[index]

    def set_value(self, value):
        """Set parameter value."""
        index = self.get_index(value)
        self.val = index

    def value(self):
        """Return parameter value."""
        return self.choices[self.index()]

    def index(self):
        """Return parameter index."""
        if self.val is None:
            self.val = self.sample(len(self.choices))
        return self.val

    def get_index(self, value):
        """Return index of value."""
        return self.choices.index(value)

    def set_index(self, index):
        """Set parameter index."""
        self.val = index

    def __len__(self):
        """Return parameter dimension."""
        if self._length is None:
            self._length = len(self.choices)
        return self._length


class ParamTensor(Param):
    """Tensor Parameter class."""

    def __init__(self, shape, space=None, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
        self.sample = _default_tensor_sampler if sampler is None else sampler
        self.shape = shape
        self.val = self.sample(self.shape)
        self._length = None

    def extra_repr(self):
        """Return extra string representation of param."""
        return 'shape={}'.format(self.shape)

    def is_valid(self, value):
        """Return True if a value is valid for this parameter."""
        return isinstance(value, torch.Tensor)

    def value(self):
        """Return parameter value."""
        if self.val is None:
            self.val = self.sample(self.shape)
        return self.val

    def set_value(self, value):
        """Set parameter value."""
        self.val = value


ArchParamSpace = ParamSpace()
