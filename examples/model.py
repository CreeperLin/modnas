import torch
import torch.nn as nn
import torch.nn.functional as F
from combo_nas.arch_space.constructor import Slot
from combo_nas.arch_space.ops import configure_ops, register_op, update_op, build_op
from combo_nas.arch_space.mixed_ops import build_mixed_op, mixed_ops
from combo_nas.core.nas_modules import NASModule
import combo_nas.arch_space as arch_space

class Prim1(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.net = nn.BatchNorm2d(C, affine=True)
        
    def forward(self, x):
        y = self.net(x)
        return y


class Prim2(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.net = nn.Sequential(
            # nn.MaxUnpool2d(2,2),
            nn.Conv2d(C, C, 3, 1, 1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.net(x)
        return y


class Prim3(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.net = nn.Sequential(
            # nn.MaxUnpool2d(2,2),
            nn.Conv2d(C, C, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        y = self.net(x)
        return y

custom_primitives = {
    'P1': lambda C_in, C_out, stride: Prim1(C_in, stride),
    'P2': lambda C_in, C_out, stride: Prim2(C_in, stride),
    'P3': lambda C_in, C_out, stride: Prim3(C_in, stride),
}

custom_ops = ['P1', 'P2', 'P3']

@mixed_ops.register('CustomMixedOp')
class CustomMixOp(NASModule):
    def __init__(self, C_in, stride, ops, shared_param):
        params_shape = (len(ops), )
        pid = -1 if shared_param else None
        super(CustomMixOp, self).__init__(params_shape, pid)
        self.ops = ops
        self._ops = nn.ModuleList([custom_primitives[n](C_in, C_in, stride) for n in ops])
    
    def param_forward(self, s):
        pass
    
    def forward(self, x):
        w = F.softmax(self.arch_param, dim=-1)
        y = sum([wi * op(x) for (wi, op) in zip(w, self._ops)])
        return y
    
    def to_genotype(self):
        w_max, prim_idx = torch.topk(self.arch_param.detach(), 1)
        gene = [self.ops[i] for i in prim_idx]
        return gene[0]
    
    def build_from_genotype(self, gene):
        pass

class CustomCell(nn.Module):
    def __init__(self, C_in, stride):
        super().__init__()
        ops = custom_ops
        edges = nn.ModuleList()
        n_edges = 5
        for i in range(n_edges):
            e_stride = stride if i==0 else 1
            edges.append(CustomMixOp(C_in, e_stride, ops, False))
        self.edges = edges
    
    def forward(self, y):
        for edge in self.edges:
            y = edge(y)
        return y


class CustomNet(nn.Module):
    def __init__(self, C_in):
        super().__init__()
        n_cells = 3
        cells = nn.ModuleList()
        for i in range(n_cells):
            cells.append(CustomCell(C_in, 1))
        self.cells = cells
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(C_in, 10)
    
    def forward(self, y):
        for cell in self.cells:
            y = cell(y)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y


class CustomBlock(nn.Module):
    def __init__(self, C_in, stride):
        super().__init__()
        ops = custom_primitives
        edges = nn.ModuleList()
        for i in range(5):
            e_stride = stride if i==0 else 1
            edges.append(Slot(C_in, C_in, e_stride, name='first_block'))
        for i in range(5):
            e_stride = stride if i==0 else 1
            edges.append(Slot(C_in, C_in, e_stride, name='last_block'))
        self.edges = edges
    
    def forward(self, y):
        for edge in self.edges:
            y = edge(y)
        return y


class CustomBackbone(nn.Module):
    def __init__(self, C_in):
        super().__init__()
        cells = nn.ModuleList()
        for i in range(3):
            cells.append(CustomBlock(C_in, 1))
        self.cells = cells
        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(C_in, 10)

    def forward(self, y):
        for cell in self.cells:
            y = cell(y)
        y = self.conv_last(y)
        y = y.view(y.size(0), -1) # flatten
        y = self.fc(y)
        return y
    
@arch_space.register('CustomNet')
def build_custom_backbone(config):
    return CustomBackbone(config.channel_in)

def custom_backbone_cvt(slot, mixed_op_cls):
    if slot.name == 'first_block':
        # ent = CustomMixOp(slot.chn_in, slot.stride, custom_ops, False)
        ent = build_mixed_op(mixed_op_cls, chn_in=slot.chn_in, chn_out=slot.chn_out, stride=slot.stride, ops=custom_ops)
    elif slot.name == 'last_block':
        ent = nn.Conv2d(slot.chn_in, slot.chn_out, 3, slot.stride, 1)
    return ent

def custom_genotype_cvt(slot, gene):
    if isinstance(gene, list): gene=gene[0]
    op_name = gene
    ent = build_op(op_name, slot.chn_in, slot.chn_out, slot.stride)
    return ent

ops_map = {
    'MAX': ['AVG', 'MAX'],
    'AVG': ['AVG', 'MAX'],
    'SC3': ['SC3', 'SC5', 'SC7'],
    'SC5': ['SC3', 'SC5', 'SC7'],
    'SC7': ['SC3', 'SC5', 'SC7'],
    'DC3': ['SC3', 'SC5', 'SC7'],
    'DC5': ['SC3', 'SC5', 'SC7'],
}

def custom_genotype_space_cvt(slot, gene, mixed_op_cls):
    idx = slot.sid
    if isinstance(gene, list): gene=gene[0]
    chn_in = slot.chn_in
    chn_out = slot.chn_out
    op_name = gene
    if op_name == 'NIL' or op_name == 'IDT':
        ent = build_op(op_name, chn_in, chn_out, slot.stride)
    else:
        ent = build_mixed_op(mixed_op_cls, chn_in=chn_in, chn_out=chn_out, stride=slot.stride, ops=ops_map[op_name])
    return ent

def register_custom_ops():
    for k, v in custom_primitives.items():
        register_op(v, k)