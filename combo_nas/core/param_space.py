import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import random

class ParamSpace():
    def __init__(self):
        self.reset()

    def reset(self, ):
        self._param_id = -1
        self._params_map = OrderedDict()
        self._categorical_length = None

    def register(self, param, name):
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
        self._params_map[reg_name] = param
        if isinstance(param, ParamCategorical):
            self._categorical_length = None

    def new_param_id(self, ):
        self._param_id += 1
        return self._param_id

    def new_param_name(self, param):
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
        for p in self._params_map.values():
            yield p

    def named_params(self, ):
        for n, p in self._params_map.items():
            yield n, p

    def get_param(self, name):
        return self._params_map.get(name, None)

    def categorical_size(self, ):
        if self._categorical_length is None:
            prod = 1
            for x in self.categorical_params():
                prod *= len(x)
            self._categorical_length = prod
        return self._categorical_length

    def categorical_params(self, ):
        for p in self.params():
            if isinstance(p, ParamCategorical):
                yield p

    def tensor_params(self, ):
        for p in self.params():
            if isinstance(p, ParamTensor):
                yield p

    def categorical_values(self, ):
        for p in self.params():
            if isinstance(p, ParamCategorical):
                yield p.value()

    def tensor_values(self, ):
        for p in self.params():
            if isinstance(p, ParamTensor):
                yield p.value()

    def get_categorical_param(self, idx):
        return list(self.categorical_params())[idx]

    def get_tensor_param(self, idx):
        return list(self.tensor_params())[idx]

    def get_categorical_value(self, idx):
        return list(self.categorical_params())[idx].value()

    def get_tensor_value(self, idx):
        return list(self.tensor_params())[idx].value()

    def get_categorical_params(self, idx):
        arch_param = OrderedDict()
        for ap in self.categorical_params():
            ap_dim = len(ap)
            arch_param[ap.name] = ap.get_value(idx % ap_dim)
            idx //= ap_dim
        return arch_param

    def get_categorical_index(self, param):
        idx = 0
        base = 1
        for n, v in param.items():
            p = self.get_param(n)
            p_dim = len(p)
            p_idx = p.get_index(v)
            idx += base * p_idx
            base *= p_dim
        return idx

    def set_categorical_params(self, idx):
        for ap in self.categorical_params():
            ap_dim = len(ap)
            ap.set_value(idx % ap_dim)
            idx //= ap_dim

    def update_params(self, pmap, trigger=True):
        for k, v in pmap.items():
            p = self.get_param(k)
            if p is None:
                logging.error('parameter \'{}\' not found'.format(k))
            p.set_value(v)
            if trigger:
                p.on_update()

    def on_update_tensor_params(self):
        for ap in self.tensor_params():
            ap.on_update()


class Param():
    def __init__(self, space, name, on_update):
        self.name = None
        self.on_update_handler = on_update
        space.register(self, name)

    def __repr__(self):
        return '{}(name={}, {})'.format(self.__class__.__name__, self.name, self.extra_repr())

    def extra_repr(self):
        return ''

    def is_valid(self, value):
        return True

    def on_update(self):
        handler = self.on_update_handler
        if not handler is None:
            return handler(self)
        return None

    def __deepcopy__(self, memo):
        # disable deepcopy
        return self


def default_categorical_sampler(dim):
    return np.random.randint(dim)


def default_tensor_sampler(shape):
    _init_ratio = 1e-3
    return nn.Parameter(_init_ratio * torch.randn(shape))


def default_int_sampler(bound):
    return random.randint(*bound)


def default_real_sampler(bound):
    return random.uniform(*bound)


class ParamNumeric(Param):
    def __init__(self, space, low, high, ntype=None, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
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


class ParamCategorical(Param):
    def __init__(self, space, choices, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
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


class ParamTensor(Param):
    def __init__(self, space, shape, sampler=None, name=None, on_update=None):
        super().__init__(space, name, on_update)
        self.sample = default_tensor_sampler if sampler is None else sampler
        self.shape = shape
        self.val = self.sample(self.shape)
        self._length = None

    def extra_repr(self):
        return 'shape={}'.format(self.shape)

    def is_valid(self, value):
        return isinstance(value, torch.Tensor)

    def value(self):
        if self.val is None:
            self.val = self.sample(self.shape)
        return self.val

    def set_value(self, value):
        self.val = value


class ArchParamSpaceClass(ParamSpace):
    def param_call(self, func, *args, **kwargs):
        return [getattr(p, func)(*args, **kwargs) for p in self.params()]

    def param_module_call(self, func, *args, **kwargs):
        for p in self.params():
            p.param_module_call(func, *args, **kwargs)

    def param_modules(self, ):
        for p in self.params():
            for m in p.modules():
                yield (p.value(), m)

    def tensor_param_modules(self, ):
        for p in self.tensor_params():
            for m in p.modules():
                yield (p.value(), m)

    def categorical_param_modules(self, ):
        for p in self.categorical_params():
            for m in p.modules():
                yield (p.value(), m)


ArchParamSpace = ArchParamSpaceClass()


class ArchParam():
    def __init__(self):
        self.pid = None
        self._modules = []

    def value(self):
        raise NotImplementedError

    def add_module(self, m):
        self._modules.append(m)
        logging.debug('param: {} add module: {}'.format(self.pid, type(m)))

    def param_module_call(self, func, *args, **kwargs):
        return [
            getattr(m, func)(self.value(), *args, **kwargs)
            for m in self.modules()]

    def module_call(self, func, *args, **kwargs):
        return [
            getattr(m, func)(*args, **kwargs)
            for m in self.modules()]

    def modules(self):
        for m in self._modules:
            yield m


class ArchParamTensor(ParamTensor, ArchParam):
    def __init__(self, shape, sampler=None, name=None, on_update=None):
        ParamTensor.__init__(self, ArchParamSpace, shape, sampler, name, on_update)
        ArchParam.__init__(self)
        logging.debug('tensor arch param {} defined: {}'.format(self.pid, self.shape))


class ArchParamCategorical(ParamCategorical, ArchParam):
    def __init__(self, choices, sampler=None, name=None, on_update=None):
        ParamCategorical.__init__(self, ArchParamSpace, choices, sampler, name, on_update)
        ArchParam.__init__(self)
        logging.debug('discrete arch param {} defined: {}'.format(self.pid, self.choices))
