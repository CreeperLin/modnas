import logging
import numpy as np
import torch
import torch.nn as nn
from .nas_modules import NASModule

class ArchParamSpace():
    _param_id = -1
    _params = []
    _discrete_length = None

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
            ArchParamSpace._discrete_length = int(np.prod([len(x) for x in ArchParamSpace.discrete_params()]))
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
    def get_discrete_param(self, idx):
        return list(self.discrete_params())[idx]
    
    @staticmethod
    def get_continuous_param(self, idx):
        return list(self.continuous_params())[idx]
       
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
    def param_forward_all():
        for ap in ArchParamSpace._params:
            ap.param_forward_all()

    @staticmethod
    def param_call(func, *args, **kwargs):
        return [getattr(p, func)(*args, **kwargs) for p in ArchParamSpace._params]
    
    @staticmethod
    def param_module_call(func, *args, **kwargs):
        for p in ArchParamSpace._params:
            p.module_call(*args, **kwargs)
    
    @staticmethod
    def param_modules():
        for p in ArchParamSpace._params:
            for m in p.modules():
                yield (p, m)
    
    # @staticmethod
    # def params_grad(loss):
    #     mmap = NASModule._modules
    #     pmap = NASModule._params_map
    #     for pid in pmap:
    #         mlist = pmap[pid]
    #         p_grad = mmap[mlist[0]].param_grad(loss)
    #         for i in range(1,len(mlist)):
    #             p_grad += mmap[mlist[i]].param_grad(loss)
    #         yield p_grad
    
    # @staticmethod
    # def param_backward(loss):
    #     mmap = NASModule._modules
    #     pmap = NASModule._params_map
    #     for pid in pmap:
    #         mlist = pmap[pid]
    #         p_grad = mmap[mlist[0]].param_grad(loss)
    #         for i in range(1,len(mlist)):
    #             p_grad += mmap[mlist[i]].param_grad(loss)
    #         NASModule._params[pid].grad = p_grad

    # @staticmethod
    # def backward_all(loss):
    #     m_out_all = []
    #     m_out_len = []
    #     for dev_id in NASModule.get_device():
    #         m_out = [m.get_state('m_out'+dev_id) for m in NASModule.modules()]
    #         m_out_all.extend(m_out)
    #         m_out_len.append(len(m_out))
    #     m_grad = torch.autograd.grad(loss, m_out_all)
    #     for i, dev_id in enumerate(NASModule.get_device()):
    #         NASModule.param_backward_from_grad(m_grad[sum(m_out_len[:i]) : sum(m_out_len[:i+1])], dev_id)

    # @staticmethod
    # def param_backward_from_grad(m_grad, dev_id):
    #     mmap = NASModule._modules
    #     pmap = NASModule._params_map
    #     for pid in pmap:
    #         mlist = pmap[pid]
    #         p_grad = 0
    #         for i in range(0,len(mlist)):
    #             p_grad = mmap[mlist[i]].param_grad_dev(m_grad[mlist[i]], dev_id) + p_grad
    #         if NASModule._params[pid].grad is None:
    #             NASModule._params[pid].grad = p_grad
    #         else:
    #             NASModule._params[pid].grad += p_grad
    
                
class ArchParam():
    def __init__(self):
        ArchParamSpace.register(self)
        self.mids = []
        self._length = None
        self.val = None
    
    def add_module(self, mid):
        if mid in self.mids: return
        self.mids.append(mid)
        print('param: {} add module: {}'.format(self.pid, mid))

    def param_forward_all(self):
        for mid in self.mids:
            self.param_forward(mid)
    
    def param_forward(self, mid):
        print('pfwd: {} {} {}'.format(self.pid, mid, self.value()))
        NASModule.get_module(mid).param_forward(self.value())
    
    def module_call(self, *args, **kwargs):
        for mid in self.mids:
            getattr(NASModule.get_module(mid), func)(self.value(), *args, **kwargs)
    
    def modules(self):
        for mid in self.mids:
            yield NASModule.get_module(mid)


class ArchParamDiscrete(ArchParam):
    def __init__(self, valrange, sampler=None):
        super().__init__()
        self.valrange = valrange
        if not sampler is None: self.sample = sampler
        logging.info('discrete arch param {} defined: {}'.format(self.pid, self.valrange))

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
        logging.info('continuous arch param {} defined: {}'.format(self.pid, self.shape))
    
    def sample(self):
        _init_ratio = 1e-3
        return nn.Parameter(_init_ratio * torch.randn(self.shape))
    
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
