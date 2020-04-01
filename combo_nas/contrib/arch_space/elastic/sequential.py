import torch
import torch.nn as nn
from .modifier import modify_attr, restore_module_attrs

def hook_module_in(module, inputs):
    if ElasticSequential.get_sequential_state(module):
        modify_attr(module, 'forward', lambda x: x)


def hook_module_out(module, inputs, result):
    restore_module_attrs(module)


class ElasticSequential():
    _module_hooks = dict()
    _sequential_state = dict()
    _groups = list()

    @staticmethod
    def add_group(group):
        ElasticSequential._groups.append(group)

    @staticmethod
    def remove_group(group):
        idx = ElasticSequential._groups.index(group)
        if not idx == -1:
            group.destroy()
            del ElasticSequential._groups[idx]

    @staticmethod
    def groups():
        for g in ElasticSequential._groups:
            yield g

    @staticmethod
    def num_groups():
        return len(ElasticSequential._groups)

    @staticmethod
    def enable_sequential_transform(module):
        if not module in ElasticSequential._module_hooks:
            h_in = module.register_forward_pre_hook(hook_module_in)
            h_out = module.register_forward_hook(hook_module_out)
            ElasticSequential._module_hooks[module] = (h_in, h_out)

    @staticmethod
    def disable_sequential_transform(module):
        if module in ElasticSequential._module_hooks:
            m_hooks = ElasticSequential._module_hooks.pop(module)
            for h in m_hooks:
                h.remove()

    @staticmethod
    def set_sequential_state(module, state):
        ElasticSequential._sequential_state[module] = state

    @staticmethod
    def reset_sequential_state(module):
        ElasticSequential._sequential_state[module] = None

    @staticmethod
    def get_sequential_state(module):
        if not module in ElasticSequential._sequential_state:
            ElasticSequential._sequential_state[module] = None
        return ElasticSequential._sequential_state[module]


class ElasticSequentialGroup():
    def __init__(self, *args):
        module_groups = []
        for m in args:
            if isinstance(m, nn.Module):
                group = [m]
            elif isinstance(m, (list, tuple)):
                group = list(m)
            else:
                raise ValueError('invalid args')
            module_groups.append(group)
        self.max_depth = len(module_groups)
        self.module_groups = module_groups
        self.enable_sequential_transform()
        ElasticSequential.add_group(self)

    def destroy(self):
        self.reset_sequential_idx()
        self.disable_sequential_transform()

    def enable_sequential_transform(self):
        for m in self.modules():
            ElasticSequential.enable_sequential_transform(m)

    def disable_sequential_transform(self):
        for m in self.modules():
            ElasticSequential.disable_sequential_transform(m)

    def set_depth_ratio(self, ratio):
        depth = int(self.max_depth * ratio)
        self.set_depth(depth)

    def set_depth(self, depth):
        if depth > len(self.module_groups):
            raise ValueError('depth out of range')
        self.set_sequential_idx(list(range(depth)), reverse=True)

    def set_sequential_idx(self, idx, reverse=False):
        if isinstance(idx, int):
            idx = [idx]
        for i, m_group in enumerate(self.module_groups):
            state = 1 if i in idx else 0
            state = 1 - state if reverse else state
            for m in m_group:
                ElasticSequential.set_sequential_state(m, state)

    def reset_sequential_idx(self):
        for m in self.modules():
            ElasticSequential.reset_sequential_state(m)

    def modules(self, active=False):
        for m_group in self.module_groups:
            for m in m_group:
                if not active or not ElasticSequential.get_sequential_state(m):
                    yield m
