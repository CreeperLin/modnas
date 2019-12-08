# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn

class NASModule(nn.Module):
    _modules = []
    _module_id = -1
    _module_state_dict = {}
    _dev_list = []

    def __init__(self, arch_param_map):
        super().__init__()
        self.id = self.get_new_id()
        self.arch_param_map = arch_param_map
        for ap in arch_param_map.values():
            ap.add_module(self.id)
        NASModule.add_module(self, self.id)
        # logging.debug('reg nas module: {} mid: {} pid: {} {}'.format(self.__class__.__name__, self.id, self.pid, params_shape))
    
    @staticmethod
    def reset():
        NASModule._modules = []
        NASModule._module_id = -1
        NASModule._module_state_dict = {}
        NASModule._dev_list = []

    @property
    def arch_param(self, name):
        return self.arch_param_map[name]

    @staticmethod
    def nasmod_state_dict():
        return {
        }
    
    @staticmethod
    def nasmod_load_state_dict(sd):
        pass

    @staticmethod
    def set_device(dev_list):
        dev_list = dev_list if len(dev_list)>0 else [None]
        NASModule._dev_list = [NASModule.get_dev_id(d) for d in dev_list]
    
    @staticmethod
    def get_device():
        return NASModule._dev_list    
    
    @staticmethod
    def get_dev_id(index):
        return 'cpu' if index is None else 'cuda:{}'.format(index)

    @staticmethod
    def get_new_id():
        NASModule._module_id += 1
        return NASModule._module_id
    
    @staticmethod
    def add_module(module, mid):
        NASModule._modules.append(module)
    
    @staticmethod
    def get_module(mid):
        return NASModule._modules[mid]
    
    @staticmethod
    def modules():
        for m in NASModule._modules:
            yield m

    @staticmethod
    def module_apply(func, **kwargs):
        return [func(m, **kwargs) for m in NASModule._modules]
    
    @staticmethod
    def module_call(func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in NASModule._modules]

    def nas_state_dict(self):
        if not self.id in NASModule._module_state_dict:
            NASModule._module_state_dict[self.id] = {}
        return NASModule._module_state_dict[self.id]
    
    def param_forward(self, *args, **kwargs):
        pass
    
    def get_state(self, name, detach=False):
        sd = self.nas_state_dict()
        if not name in sd: return
        ret = sd[name]
        if detach:
            return ret.detach()
        else:
            return ret
    
    def set_state(self, name, val, detach=False):
        if detach:
            self.nas_state_dict()[name] = val.detach()
        else:
            self.nas_state_dict()[name] = val
    
    def del_state(self, name):
        if not name in self.nas_state_dict(): return
        del self.nas_state_dict()[name]
    
    @staticmethod
    def build_from_genotype_all(gene, *args, **kwargs):
        if gene.ops is None: return
        assert len(NASModule._modules) == len(gene.ops)
        for m, g in zip(NASModule._modules, gene.ops):
            m.build_from_genotype(g, *args, **kwargs)
    
    @staticmethod
    def to_genotype_all(*args, **kwargs):
        gene = []
        for m in NASModule._modules:
            _, g_module = m.to_genotype(*args, **kwargs)
            gene.append(g_module)
        return gene

    def build_from_genotype(self, gene, *args, **kwargs):
        raise NotImplementedError
    
    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError

