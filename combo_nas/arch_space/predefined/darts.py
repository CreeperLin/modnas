# -*- coding: utf-8 -*-
import torch.nn as nn
from ..layers import build as build_layer
from ..ops import FactorizedReduce, StdConv
from ..constructor import Slot, default_mixed_op_converter

class PreprocLayer(StdConv):
    def __init__(self, C_in, C_out):
        super(PreprocLayer, self).__init__(C_in, C_out, 1, 1, 0)


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class DARTSLikeNet(nn.Module):
    def __init__(self, chn_in, chn, n_classes, n_inputs_model, n_inputs_layer, n_inputs_node,
                layers, shared_a, channel_multiplier, auxiliary, cell_cls, cell_kwargs):
        super().__init__()
        self.chn_in = chn_in
        self.chn = chn
        assert n_inputs_model == 1
        assert n_inputs_layer == 2
        assert n_inputs_node == 1
        self.n_inputs_model = n_inputs_model
        self.n_inputs_layer = n_inputs_layer
        self.n_inputs_node = n_inputs_node
        self.aux_pos = 2*layers//3 if auxiliary else -1
        self.shared_a = shared_a

        chn_cur = self.chn * channel_multiplier
        self.conv_first = nn.Sequential(
            nn.Conv2d(self.chn_in, chn_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chn_cur),
        )

        chn_pp, chn_p, chn_cur = chn_cur, chn_cur, self.chn

        self.cells = nn.ModuleList()
        self.cell_group = [[],[] if shared_a else []]
        reduction_p = False
        for i in range(layers):
            stride = 1
            cell_kwargs['preproc'] = (PreprocLayer, PreprocLayer)
            if i in [layers//3, 2*layers//3]:
                reduction = True
                stride = 2
                chn_cur *= 2
            else:
                reduction = False
            if reduction_p:
                cell_kwargs['preproc'] = (FactorizedReduce, PreprocLayer)
            cell_kwargs['chn_in'] = (chn_pp, chn_p)
            cell_kwargs['stride'] = stride
            cell_kwargs['name'] = 'reduce' if reduction else 'normal'
            cell_kwargs['edge_kwargs']['chn_in'] = (chn_cur, )
            cell = build_layer(cell_cls, **cell_kwargs)
            self.cells.append(cell)
            self.cell_group[1 if reduction else 0].append(cell)
            chn_out = chn_cur * cell_kwargs['n_nodes']
            chn_pp, chn_p = chn_p, chn_out
            reduction_p = reduction
            if i == self.aux_pos:
                fm_size = 32
                self.aux_head = AuxiliaryHead(fm_size//4, chn_p, n_classes)

        self.conv_last = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(chn_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.conv_first(x)
        for cell in self.cells:
            s0, s1 = s1, cell([s0, s1])
        y = self.conv_last(s1)
        y = y.view(y.size(0), -1) # flatten
        return self.fc(y)

    def forward_aux(self, x):
        if not self.training or self.aux_pos == -1:
            return None
        s0 = s1 = self.conv_first(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell([s0, s1])
            if i == self.aux_pos:
                return self.aux_head(s1)

    def build_from_genotype(self, gene, *args, **kwargs):
        assert len(self.cell_group) == len(gene.dag)
        for cells, g in zip(self.cell_group, gene.dag):
            for c in cells:
                c.build_from_genotype(g, *args, **kwargs)

    def to_genotype(self, k=2):
        gene = []
        for cells in self.cell_group:
            gene.append(cells[0].to_genotype(k))
        return gene

    def dags(self):
        for cell in self.cells:
            yield cell

    def get_predefined_search_converter(self):
        if not self.shared_a: return default_mixed_op_converter
        def convert_fn(slot, *args, **kwargs):
            if not hasattr(convert_fn, 'param_map'):
                convert_fn.param_map = {}
            arch_params = convert_fn.param_map.get(slot.name, None)
            if not arch_params is None:
                if 'mixed_op_args' not in kwargs:
                    kwargs['mixed_op_args'] = {}
                kwargs['mixed_op_args']['arch_param_map'] = arch_params
            ent = default_mixed_op_converter(slot, *args, **kwargs)
            if not slot.name in convert_fn.param_map:
                convert_fn.param_map[slot.name] = ent.arch_param_map
            return ent
        return convert_fn

    def get_predefined_augment_converter(self):
        return lambda slot: nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(slot.chn_in, slot.chn_out, 3, slot.stride, 1, bias=False),
            nn.BatchNorm2d(slot.chn_out),
        )


def build_from_config(darts_cls=DARTSLikeNet, **kwargs):
    n_nodes = 4
    if 'nodes' in kwargs:
        n_nodes = kwargs.pop('nodes')
    darts_kwargs = {
        'n_inputs_model': 1,
        'n_inputs_layer': 2,
        'n_inputs_node': 1,
        'cell_cls': 'DAG',
        'cell_kwargs': {
            'chn_in': None,
            'chn_out': None,
            'stride': None,
            'n_nodes': n_nodes,
            'allocator': 'ReplicateAllocator',
            'merger_state': 'SumMerger',
            'merger_out': 'ConcatMerger',
            'enumerator': 'CombinationEnumerator',
            'preproc': None,
            'edge_cls': Slot,
            'edge_kwargs': {
                'chn_in': None,
                'chn_out': None,
                'stride': None,
            },
        },
    }
    darts_kwargs.update(kwargs)
    return darts_cls(**darts_kwargs)
