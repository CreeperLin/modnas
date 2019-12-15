# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn

class ArchModuleSpace():
    _modules = []
    _module_id = -1
    _module_state_dict = {}
    _params_map = []
    _dev_list = []

    @staticmethod
    def reset():
        ArchModuleSpace._modules = []
        ArchModuleSpace._module_id = -1
        ArchModuleSpace._module_state_dict = {}
        ArchModuleSpace._dev_list = []
        ArchModuleSpace._params_map = []

    @staticmethod
    def nasmod_state_dict():
        return {
        }
    
    @staticmethod
    def nasmod_load_state_dict(sd):
        pass

    @staticmethod
    def get_new_id():
        ArchModuleSpace._module_id += 1
        return ArchModuleSpace._module_id
    
    @staticmethod
    def register(module, mid, arch_param_map):
        ArchModuleSpace._modules.append(module)
        ArchModuleSpace._params_map.append(arch_param_map)
    
    @staticmethod
    def get_module(mid):
        return ArchModuleSpace._modules[mid]
    
    @staticmethod
    def modules():
        for m in ArchModuleSpace._modules:
            yield m

    @staticmethod
    def module_apply(func, **kwargs):
        return [func(m, **kwargs) for m in ArchModuleSpace._modules]
    
    @staticmethod
    def module_call(func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in ArchModuleSpace._modules]
    
    @staticmethod
    def to_genotype_all(*args, **kwargs):
        gene = []
        for m in ArchModuleSpace._modules:
            _, g_module = m.to_genotype(*args, **kwargs)
            gene.append(g_module)
        return gene

    def build_from_genotype(self, gene, *args, **kwargs):
        raise NotImplementedError
    
    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError


class NASModule(nn.Module):
    def __init__(self, arch_param_map):
        super().__init__()
        self.id = ArchModuleSpace.get_new_id()
        for ap in arch_param_map.values():
            ap.add_module(self.id)
        ArchModuleSpace.register(self, self.id, arch_param_map)
        # logging.debug('reg nas module: {} mid: {} p: {}'.format(self.__class__.__name__, self.id, arch_param_map))
    
    def arch_param(self, name):
        return (ArchModuleSpace._params_map[self.id]).get(name)
    
    def arch_param_value(self, name):
        return (ArchModuleSpace._params_map[self.id]).get(name).value()
    
    def arch_param_map(self):
        return ArchModuleSpace._params_map[self.id]

    def nas_state_dict(self):
        if not self.id in ArchModuleSpace._module_state_dict:
            ArchModuleSpace._module_state_dict[self.id] = {}
        return ArchModuleSpace._module_state_dict[self.id]

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