# -*- coding: utf-8 -*-
import logging
import torch.nn as nn

class ArchModuleSpaceClass():
    def __init__(self, ):
        self.reset()

    def reset(self, ):
        self._modules = []
        self._module_id = -1
        self._params_map = []

    def state_dict(self, ):
        return {
        }

    def load_state_dict(self, sd):
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

    def params_map(self, pid):
        if pid < len(self._params_map):
            return self._params_map[pid]
        else:
            raise ValueError('invalid pid: {}'.format(pid))

    def module_apply(self, func, **kwargs):
        return [func(m, **kwargs) for m in self.modules()]

    def module_call(self, func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in self.modules()]

    def to_genotype_all(self, *args, **kwargs):
        gene = []
        for m in self.modules():
            gene.append(m.to_genotype(*args, **kwargs))
        return gene


ArchModuleSpace = ArchModuleSpaceClass()


class NASModule(nn.Module):
    def __init__(self, arch_param_map):
        super().__init__()
        ArchModuleSpace.register(self, arch_param_map)
        for ap in arch_param_map.values():
            ap.add_module(self.id)
        logging.debug('reg nas module: {} mid: {} p: {}'.format(self.__class__.__name__, self.id, arch_param_map))

    def arch_param(self, name):
        return ArchModuleSpace.params_map(self.id).get(name)

    def arch_param_value(self, name):
        return ArchModuleSpace.params_map(self.id).get(name).value()

    def arch_param_map(self):
        return ArchModuleSpace.params_map(self.id)

    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError
