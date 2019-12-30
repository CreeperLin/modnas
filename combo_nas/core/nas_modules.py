# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn

class ArchModuleSpaceClass():
    def __init__(self, ):
        self.reset()

    def reset(self, ):
        self._modules = []
        self._module_id = -1
        self._module_state_dict = {}
        self._params_map = []

    def nasmod_state_dict(self, ):
        return {
        }
    
    def nasmod_load_state_dict(self, sd):
        pass

    def get_new_id(self, ):
        self._module_id += 1
        return self._module_id
    
    def register(self, module, arch_param_map):
        module.id = self.get_new_id()
        self._modules.append(module)
        self._params_map.append(arch_param_map)
    
    def get_module(self, mid):
        return self._modules[mid]
    
    def modules(self, ):
        for m in self._modules:
            yield m

    def module_apply(self, func, **kwargs):
        return [func(m, **kwargs) for m in self.modules()]
    
    def module_call(self, func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in self.modules()]
    
    def to_genotype_all(self, *args, **kwargs):
        gene = []
        for m in self.modules():
            _, g_module = m.to_genotype(*args, **kwargs)
            gene.append(g_module)
        return gene

ArchModuleSpace = ArchModuleSpaceClass()

class NASModule(nn.Module):
    def __init__(self, arch_param_map):
        super().__init__()
        ArchModuleSpace.register(self, arch_param_map)
        for ap in arch_param_map.values():
            ap.add_module(self.id)
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
    
    def build_from_genotype(self, gene, *args, **kwargs):
        raise NotImplementedError
    
    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError
