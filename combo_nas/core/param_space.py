import logging
import numpy as np
import torch
import torch.nn as nn
from .nas_modules import ArchModuleSpace

class ArchParamSpace():
    _param_id = -1
    _params = []
    _discrete_length = None

    @staticmethod
    def reset():
        ArchParamSpace._param_id = -1
        ArchParamSpace._params = []
        ArchParamSpace._discrete_length = None

    @staticmethod
    def register(param):
        ArchParamSpace._params.append(param)
        param.pid = ArchParamSpace.new_param_id()
    
    @staticmethod
    def new_param_id():
        ArchParamSpace._param_id += 1
        return ArchParamSpace._param_id
    
    @staticmethod
    def params():
        for p in ArchParamSpace._params:
            yield p
    
    @staticmethod
    def discrete_size():
        if ArchParamSpace._discrete_length is None:
            prod = 1
            for x in ArchParamSpace.discrete_params():
                prod *= len(x)
            ArchParamSpace._discrete_length = prod
        return ArchParamSpace._discrete_length
    
    @staticmethod
    def discrete_params():
        for p in ArchParamSpace._params:
            if isinstance(p, ArchParamDiscrete):
                yield p
    
    @staticmethod
    def continuous_params():
        for p in ArchParamSpace._params:
            if isinstance(p, ArchParamContinuous):
                yield p

    @staticmethod
    def discrete_values():
        for p in ArchParamSpace._params:
            if isinstance(p, ArchParamDiscrete):
                yield p.value()
    
    @staticmethod
    def continuous_values():
        for p in ArchParamSpace._params:
            if isinstance(p, ArchParamContinuous):
                yield p.value()

    @staticmethod
    def get_discrete_param(self, idx):
        return list(self.discrete_params())[idx]
    
    @staticmethod
    def get_continuous_param(self, idx):
        return list(self.continuous_params())[idx]
       
    @staticmethod
    def get_discrete_value(self, idx):
        return list(self.discrete_params())[idx].value()
    
    @staticmethod
    def get_continuous_value(self, idx):
        return list(self.continuous_params())[idx].value()

    @staticmethod
    def get_discrete(idx):
        arch_param = []
        for ap in ArchParamSpace.discrete_params():
            ap_dim = len(ap)
            arch_param.append(ap.get_value(idx % ap_dim))
            idx //= ap_dim
        return arch_param

    @staticmethod
    def set_discrete(idx):
        for ap in ArchParamSpace.discrete_params():
            ap_dim = len(ap)
            ap.set_value(idx % ap_dim)
            idx //= ap_dim

    @staticmethod
    def param_call(func, *args, **kwargs):
        return [getattr(p, func)(*args, **kwargs) for p in ArchParamSpace._params]
    
    @staticmethod
    def param_module_call(func, *args, **kwargs):
        for p in ArchParamSpace._params:
            p.param_module_call(func, *args, **kwargs)
    
    @staticmethod
    def param_modules():
        for p in ArchParamSpace._params:
            for m in p.modules():
                yield (p.value(), m)

    @staticmethod
    def continuous_param_modules():
        for p in ArchParamSpace.continuous_params():
            for m in p.modules():
                yield (p.value(), m)

    @staticmethod
    def discrete_param_modules():
        for p in ArchParamSpace.discrete_params():
            for m in p.modules():
                yield (p.value(), m)

    @staticmethod
    def backward_all(loss):
        m_out_all = []
        m_out_len = []
        for dev_id in ArchModuleSpace.get_device():
            m_out = [m.get_state('m_out'+dev_id) for m in ArchModuleSpace.modules()]
            m_out_all.extend(m_out)
            m_out_len.append(len(m_out))
        m_grad = torch.autograd.grad(loss, m_out_all)
        for i, dev_id in enumerate(ArchModuleSpace.get_device()):
            ArchParamSpace.param_backward_from_grad(m_grad[sum(m_out_len[:i]) : sum(m_out_len[:i+1])], dev_id)

    @staticmethod
    def param_backward_from_grad(m_grad, dev_id):
        pmap = ArchParamSpace.continuous_params()
        for p in pmap:
            p_grad = sum([
                ArchModuleSpace.get_module(mid).param_grad_dev(m_grad[mid], dev_id)
                for mid in p.mids])
            pv = p.value()
            if pv.grad is None:
                pv.grad = p_grad
            else:
                pv.grad += p_grad
    
                
class ArchParam():
    def __init__(self):
        ArchParamSpace.register(self)
        self.mids = []
        self._length = None
        self.val = None
    
    def add_module(self, mid):
        if mid in self.mids: return
        self.mids.append(mid)
        logging.debug('param: {} add module: {}'.format(self.pid, mid))
    
    def param_module_call(self, func, *args, **kwargs):
        return [
            getattr(ArchModuleSpace.get_module(mid), func)(self.value(), *args, **kwargs)
        for mid in self.mids]
    
    def module_call(self, func, *args, **kwargs):
        return [
            getattr(ArchModuleSpace.get_module(mid), func)(*args, **kwargs)
        for mid in self.mids]

    def modules(self):
        for mid in self.mids:
            yield ArchModuleSpace.get_module(mid)


class ArchParamDiscrete(ArchParam):
    def __init__(self, valrange, sampler=None):
        super().__init__()
        self.valrange = valrange
        if not sampler is None: self.sample = sampler
        logging.debug('discrete arch param {} defined: {}'.format(self.pid, self.valrange))

    def sample(self):
        return self.valrange[np.random.randint(len(self))]

    def get_value(self, index):
        return self.valrange[index]
    
    def set_value(self, index):
        self.val = self.valrange[index]
    
    def value(self):
        if self.val is None:
            self.val = self.sample()
        return self.val
    
    def get_index(self, value):
        return self.valrange.index(value)
    
    def __len__(self):
        if self._length is None:
            self._length = len(self.valrange)
        return self._length


class ArchParamContinuous(ArchParam):

    def __init__(self, shape, sampler=None):
        super().__init__()
        self.shape = shape
        self.val = self.sample()
        if not sampler is None: self.sample = sampler
        logging.debug('continuous arch param {} defined: {}'.format(self.pid, self.shape))
    
    def sample(self):
        _init_ratio = 1e-3
        val = nn.Parameter(_init_ratio * torch.randn(self.shape))
        return val
    
    def value(self):
        if self.val is None:
            self.val = self.sample()
        return self.val

    def get_value(self):
        return self.val
    
    def set_value(self, value):
        self.val = value

    def __len__(self):
        return self._length
