import torch
import torch.nn as nn
from .modifier import modify_param, modify_buffer, modify_attr,\
    restore_module_states, get_ori_buffer

def conv2d_fan_out_trnsf(m, idx):
    modify_param(m, 'weight', m.weight[idx, :, :, :])
    if not m.bias is None:
        modify_param(m, 'bias', m.bias[idx])
    if m.groups != 1 and m.weight.shape[1] == 1:
        width = idx.stop - idx.start if isinstance(idx, slice) else len(idx)
        modify_attr(m, 'groups', width)


def conv2d_fan_in_trnsf(m, idx):
    bias_idx = None
    if m.groups == 1:
        modify_param(m, 'weight', m.weight[:, idx, :, :])
    elif m.weight.shape[1] == 1:
        width = idx.stop - idx.start if isinstance(idx, slice) else len(idx)
        modify_param(m, 'weight', m.weight[idx, :, :, :])
        modify_attr(m, 'groups', width)
        bias_idx = idx
    else:
        raise NotImplementedError
    if not m.bias is None and not bias_idx is None:
        modify_param(m, 'bias', m.bias[bias_idx])


def batchnorm2d_fan_in_out_trnsf(m, idx):
    if not m.weight is None:
        modify_param(m, 'weight', m.weight[idx])
    if not m.bias is None:
        modify_param(m, 'bias', m.bias[idx])
    modify_buffer(m, 'running_mean', m.running_mean[idx])
    modify_buffer(m, 'running_var', m.running_var[idx])


def batchnorm2d_fan_in_out_post_trnsf(m, idx):
    if isinstance(idx, slice):
        return
    get_ori_buffer(m, 'running_mean')[idx] = m.running_mean
    get_ori_buffer(m, 'running_var')[idx] = m.running_var


_fan_out_transform = {
    nn.Conv2d: conv2d_fan_out_trnsf,
    nn.BatchNorm2d: batchnorm2d_fan_in_out_trnsf,
}

_fan_in_transform = {
    nn.Conv2d: conv2d_fan_in_trnsf,
    nn.BatchNorm2d: batchnorm2d_fan_in_out_trnsf,
}

_fan_out_post_transform = {
    nn.BatchNorm2d: batchnorm2d_fan_in_out_post_trnsf,
}

_fan_in_post_transform = {
    nn.BatchNorm2d: batchnorm2d_fan_in_out_post_trnsf,
}


def get_fan_out_transform(mtype):
    return _fan_out_transform.get(mtype, None)


def get_fan_in_transform(mtype):
    return _fan_in_transform.get(mtype, None)


def get_fan_out_post_transform(mtype):
    return _fan_out_post_transform.get(mtype, None)


def get_fan_in_post_transform(mtype):
    return _fan_in_post_transform.get(mtype, None)


def set_fan_out_transform(mtype, transf):
    _fan_out_transform[mtype] = transf


def set_fan_in_transform(mtype, transf):
    _fan_in_transform[mtype] = transf


def set_fan_out_post_transform(mtype, transf):
    _fan_out_post_transform[mtype] = transf


def set_fan_in_post_transform(mtype, transf):
    _fan_in_post_transform[mtype] = transf


def hook_module_in(module, inputs):
    # print('hin', module.__class__.__name__, inputs[0].shape)
    fan_in_idx, fan_out_idx = ElasticSpatial.get_spatial_idx(module)
    mtype = type(module)
    trnsf = get_fan_in_transform(mtype)
    if not trnsf is None and not fan_in_idx is None:
        trnsf(module, fan_in_idx)
    trnsf = get_fan_out_transform(mtype)
    if not trnsf is None and not fan_out_idx is None:
        trnsf(module, fan_out_idx)


def hook_module_out(module, inputs, outputs):
    # print('hout', module.__class__.__name__, outputs[0].shape)
    fan_in_idx, fan_out_idx = ElasticSpatial.get_spatial_idx(module)
    mtype = type(module)
    trnsf = get_fan_in_post_transform(mtype)
    if not trnsf is None and not fan_in_idx is None:
        trnsf(module, fan_in_idx)
    trnsf = get_fan_out_post_transform(mtype)
    if not trnsf is None and not fan_out_idx is None:
        trnsf(module, fan_out_idx)
    restore_module_states(module)


class ElasticSpatial():
    _module_hooks = dict()
    _groups = list()

    @staticmethod
    def add_group(group):
        ElasticSpatial._groups.append(group)

    @staticmethod
    def remove_group(group):
        idx = ElasticSpatial._groups.index(group)
        if not idx == -1:
            group.destroy()
            del ElasticSpatial._groups[idx]

    @staticmethod
    def groups():
        for g in ElasticSpatial._groups:
            yield g

    @staticmethod
    def num_groups():
        return len(ElasticSpatial._groups)

    @staticmethod
    def enable_spatial_transform(module):
        if not module in ElasticSpatial._module_hooks:
            h_in = module.register_forward_pre_hook(hook_module_in)
            h_out = module.register_forward_hook(hook_module_out)
            ElasticSpatial._module_hooks[module] = (h_in, h_out)

    @staticmethod
    def disable_spatial_transform(module):
        if module in ElasticSpatial._module_hooks:
            m_hooks = ElasticSpatial._module_hooks.pop(module)
            for h in m_hooks:
                h.remove()
            del module._spatial_idx

    @staticmethod
    def set_spatial_fan_in_idx(module, idx):
        ElasticSpatial.get_spatial_idx(module)[0] = idx

    @staticmethod
    def set_spatial_fan_out_idx(module, idx):
        ElasticSpatial.get_spatial_idx(module)[1] = idx

    @staticmethod
    def reset_spatial_fan_in_idx(module):
        ElasticSpatial.get_spatial_idx(module)[0] = None

    @staticmethod
    def reset_spatial_fan_out_idx(module):
        ElasticSpatial.get_spatial_idx(module)[1] = None

    @staticmethod
    def reset_spatial_idx(module):
        module._spatial_idx = [None, None]

    @staticmethod
    def get_spatial_idx(module):
        if not hasattr(module, '_spatial_idx'):
            module._spatial_idx = [None, None]
        return module._spatial_idx

    @staticmethod
    def set_spatial_idx(module, fan_in, fan_out):
        module._spatial_idx = [fan_in, fan_out]


class ElasticSpatialGroup():
    def __init__(self, fan_out_modules, fan_in_modules, max_width=None, rank_fn=None):
        super().__init__()
        if fan_in_modules is None:
            fan_in_modules = []
        if fan_out_modules is None:
            fan_out_modules = []
        self.fan_out_modules = fan_out_modules
        self.fan_in_modules = fan_in_modules
        self.idx_mapping = dict()
        self.rank_fn = rank_fn
        self.cur_rank = None
        self.max_width = max_width
        self.enable_spatial_transform()
        ElasticSpatial.add_group(self)

    def destroy(self):
        self.reset_spatial_idx()
        self.disable_spatial_transform()

    def add_fan_in_module(self, module):
        self.fan_in_modules.append(module)

    def add_fan_out_module(self, module):
        self.fan_out_modules.append(module)

    def add_idx_mapping(self, dest, map_fn):
        self.idx_mapping[dest] = map_fn

    def map_index(self, idx, dest):
        map_fn = self.idx_mapping.get(dest, None)
        if map_fn is None:
            return idx
        return [map_fn(i) for i in idx]

    def enable_spatial_transform(self):
        for m in self.fan_in_modules + self.fan_out_modules:
            ElasticSpatial.enable_spatial_transform(m)

    def disable_spatial_transform(self):
        for m in self.fan_in_modules + self.fan_out_modules:
            ElasticSpatial.disable_spatial_transform(m)

    def set_width_ratio(self, ratio, rank=None):
        if ratio is None:
            self.reset_spatial_idx()
            return
        if self.max_width is None:
            raise ValueError('max_width not specified')
        width = int(self.max_width * ratio)
        self.set_width(width, rank)

    def set_width(self, width, rank=None):
        if width is None:
            self.reset_spatial_idx()
            return
        if self.cur_rank is None:
            self.set_spatial_rank()
        rank = self.cur_rank
        if rank is None:
            idx = slice(0, width)
        else:
            idx = rank[:width]
        self.set_spatial_idx(idx)

    def reset_spatial_rank(self):
        self.cur_rank = None

    def set_spatial_rank(self, rank=None):
        if rank is None and not self.rank_fn is None:
            rank = self.rank_fn()
        self.cur_rank = rank

    def set_spatial_idx(self, idx):
        if idx is None:
            self.reset_spatial_idx()
            return
        if isinstance(idx, int):
            idx = [idx]
        for m in self.fan_in_modules:
            m_idx = self.map_index(idx, m)
            ElasticSpatial.set_spatial_fan_in_idx(m, m_idx)
        for m in self.fan_out_modules:
            m_idx = self.map_index(idx, m)
            ElasticSpatial.set_spatial_fan_out_idx(m, m_idx)

    def reset_spatial_idx(self):
        for m in self.fan_in_modules:
            ElasticSpatial.reset_spatial_fan_in_idx(m)
        for m in self.fan_out_modules:
            ElasticSpatial.reset_spatial_fan_out_idx(m)


def conv2d_rank_weight_l1norm_fan_in(module):
    if module.groups == 1:
        sum_dim = 0
    elif module.weight.shape[1] == 1:
        sum_dim = 1
    else:
        raise NotImplementedError
    _, idx = torch.sort(torch.sum(torch.abs(module.weight.data), dim=(sum_dim, 2, 3)), dim=0, descending=True)
    return idx


def conv2d_rank_weight_l1norm_fan_out(module):
    _, idx = torch.sort(torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3)), dim=0, descending=True)
    return idx


def batchnorm2d_rank_weight_l1norm(module):
    _, idx = torch.sort(torch.abs(module.weight.data), dim=0, descending=True)
    return idx
