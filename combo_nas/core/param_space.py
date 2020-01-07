import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

class ParamSpace():
    def __init__(self):
        self.reset()

    def reset(self, ):
        self._param_id = -1
        self._params_map = OrderedDict()
        self._discrete_length = None

    def register(self, param, name):
        param.pid = self.new_param_id()
        if name is None:
            name = self.new_param_name(param)
        param.name = name
        self._params_map[name] = param
        if isinstance(param, ParamDiscrete):
            self._discrete_length = None

    def new_param_id(self, ):
        self._param_id += 1
        return self._param_id

    def new_param_name(self, param):
        init = 'c' if isinstance(param, ParamContinuous) else 'd'
        return '{}_{}'.format(init, self._param_id)

    def params(self, ):
        for p in self._params_map.values():
            yield p

    def get_param(self, name):
        return self._params_map.get(name, None)

    def discrete_size(self, ):
        if self._discrete_length is None:
            prod = 1
            for x in self.discrete_params():
                prod *= len(x)
            self._discrete_length = prod
        return self._discrete_length

    def discrete_params(self, ):
        for p in self.params():
            if isinstance(p, ParamDiscrete):
                yield p

    def continuous_params(self, ):
        for p in self.params():
            if isinstance(p, ParamContinuous):
                yield p

    def discrete_values(self, ):
        for p in self.params():
            if isinstance(p, ParamDiscrete):
                yield p.value()

    def continuous_values(self, ):
        for p in self.params():
            if isinstance(p, ParamContinuous):
                yield p.value()

    def get_discrete_param(self, idx):
        return list(self.discrete_params())[idx]

    def get_continuous_param(self, idx):
        return list(self.continuous_params())[idx]

    def get_discrete_value(self, idx):
        return list(self.discrete_params())[idx].value()

    def get_continuous_value(self, idx):
        return list(self.continuous_params())[idx].value()

    def get_discrete_map(self, idx):
        arch_param = {}
        for ap in self.discrete_params():
            ap_dim = len(ap)
            arch_param[ap.name] = ap.get_value(idx % ap_dim)
            idx //= ap_dim
        return arch_param

    def set_discrete_params(self, idx):
        for ap in self.discrete_params():
            ap_dim = len(ap)
            ap.set_value(idx % ap_dim)
            idx //= ap_dim

    def set_params_map(self, pmap):
        for k, v in pmap.items():
            self.get_param(k).set_value(v)


class Param():
    def __init__(self, space, name):
        space.register(self, name)
    
    def __deepcopy__(self, memo):
        # disable deepcopy
        return self


def default_discrete_sampler(dim):
    return np.random.randint(dim)


def default_continuous_sampler(shape):
    _init_ratio = 1e-3
    return nn.Parameter(_init_ratio * torch.randn(shape))


class ParamDiscrete():
    def __init__(self, valrange, sampler=None):
        self.sample = default_discrete_sampler if sampler is None else sampler
        self.valrange = valrange
        self._length = None
        self.val = None

    def get_value(self, index):
        return self.valrange[index]

    def set_value(self, value):
        index = self.get_index(value)
        self.val = index

    def value(self):
        return self.valrange[self.index()]

    def index(self):
        if self.val is None:
            self.val = self.sample(len(self.valrange))
        return self.val

    def get_index(self, value):
        return self.valrange.index(value)

    def set_index(self, index):
        self.val = index

    def __len__(self):
        if self._length is None:
            self._length = len(self.valrange)
        return self._length


class ParamContinuous():
    def __init__(self, shape, sampler):
        self.sample = default_continuous_sampler if sampler is None else sampler
        self.shape = shape
        self.val = self.sample(self.shape)
        self._length = None

    def value(self):
        if self.val is None:
            self.val = self.sample(self.shape)
        return self.val

    def get_value(self):
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

    def continuous_param_modules(self, ):
        for p in self.continuous_params():
            for m in p.modules():
                yield (p.value(), m)

    def discrete_param_modules(self, ):
        for p in self.discrete_params():
            for m in p.modules():
                yield (p.value(), m)


ArchParamSpace = ArchParamSpaceClass()


class ArchParam(Param):
    def __init__(self, name):
        self.pid = None
        super().__init__(ArchParamSpace, name)
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


class ArchParamContinuous(ParamContinuous, ArchParam):
    def __init__(self, shape, sampler=None, name=None):
        ParamContinuous.__init__(self, shape, sampler)
        ArchParam.__init__(self, name)
        logging.debug('continuous arch param {} defined: {}'.format(self.pid, self.shape))


class ArchParamDiscrete(ParamDiscrete, ArchParam):
    def __init__(self, valrange, sampler=None, name=None):
        ParamDiscrete.__init__(self, valrange, sampler)
        ArchParam.__init__(self, name)
        logging.debug('discrete arch param {} defined: {}'.format(self.pid, self.valrange))
