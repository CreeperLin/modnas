import logging
import torch
import torch.nn as nn
from .ops import build_op, Identity, DropPath_
from .mixed_ops import build_mixed_op
from .layers import build_layer
from . import genotypes as gt

class Slot(nn.Module):
    _slots = []
    _slot_id = -1
    _convert_fn = None

    def __init__(self, chn_in, chn_out, stride, name=None, arch_param_map=None, *args, **kwargs):
        super().__init__()
        Slot.register(self)
        self.name = str(self.sid) if name is None else name
        self.arch_param_map = arch_param_map
        self.e_chn_in = chn_in
        self.e_chn_out = chn_in if chn_out is None else chn_out
        self.stride = stride
        self.ent = None
        self.gene = None
        self.args = args
        self.kwargs = kwargs
        self.fixed = False
        self.visited = False
        logging.debug('slot {} {} {}: declared {} {} {}'.format(
            self.sid, self.arch_param_map, self.name, self.chn_in, self.chn_out, self.stride))
    
    @staticmethod
    def register(slot):
        slot.sid = Slot.new_slot_id()
        Slot._slots.append(slot)
    
    @staticmethod
    def reset():
        Slot._slots = []
        Slot._slot_id = -1
        Slot._convert_fn = None

    @staticmethod
    def new_slot_id():
        Slot._slot_id += 1
        return Slot._slot_id
    
    @property
    def chn_in(self):
        chn_in = self.e_chn_in
        return chn_in[0] if isinstance(chn_in, list) and len(chn_in) == 1 else chn_in
    
    @property
    def chn_out(self):
        chn_out = self.e_chn_out
        return chn_out[0] if isinstance(chn_out, list) and len(chn_out) == 1 else chn_out
        
    @staticmethod
    def slots_all():
        for m in Slot._slots:
            yield m
    
    @staticmethod
    def to_genotype_all(*args, **kwargs):
        gene = []
        for m in Slot._slots:
            if m.visited: continue
            _, g = m.to_genotype(*args, **kwargs)
            gene.append(g)
        for m in Slot._slots:
            m.visited = False
        return gene

    def set_entity(self, ent):
        if self.fixed: return
        self.ent = ent
        self.forward = ent.forward
        self.__call__ = ent.__call__
        logging.debug('slot {} {}: set to {}'.format(self.sid, self.name, ent.__class__.__name__))
    
    def forward(self, x):
        raise ValueError('Undefined entity in slot {}'.format(self.sid))

    def to_genotype(self, *args, **kwargs):
        self.visited = True
        if hasattr(self.ent, 'to_genotype'):
            return self.ent.to_genotype(*args, **kwargs)
        else:
            logging.debug('slot {} default genotype {}'.format(self.sid, self.gene))
            return 1, self.gene

    def build_from_genotype(self, gene, *args, **kwargs):
        self.gene = gene
        if self.ent is None:
            convert_fn = default_genotype_converter if Slot._convert_fn is None else Slot._convert_fn
            ent = convert_fn(self, gene, *args, **kwargs)
            self.set_entity(ent)
        else:
            self.ent.build_from_genotype(gene, *args, **kwargs)
        self.built = True


def default_predefined_converter(slot, mixed_op_cls, *args, **kwargs):
    ent = build_mixed_op(mixed_op_cls, 
                        chn_in=slot.chn_in, 
                        chn_out=slot.chn_out, 
                        stride=slot.stride, 
                        ops=gt.get_primitives(), 
                        *args, **kwargs)
    return ent

def apply_drop_path(ent, drop_path):
    if drop_path and not isinstance(ent, Identity):
        ent = nn.Sequential(
            ent,
            DropPath_()
        )
    return ent

def default_genotype_converter(slot, gene):
    if isinstance(gene, list): gene = gene[0]
    op_name = gene
    ent = build_op(op_name, slot.chn_in, slot.chn_out, slot.stride)
    return ent

def convert_from_predefined_net(model, convert_fn=None, drop_path=False, *args, **kwargs):
    convert_fn = default_predefined_converter if convert_fn is None else convert_fn
    logging.info('convert from predefined net')
    logging.debug('converter: {}'.format(convert_fn.__qualname__))
    for m in Slot.slots_all():
        if m.fixed: continue
        ent = apply_drop_path(convert_fn(m, *args, **kwargs), drop_path)
        m.set_entity(ent)
    return model

def convert_from_genotype(model, genotype, convert_fn=None, drop_path=False, *args, **kwargs):
    convert_fn = default_genotype_converter if convert_fn is None else convert_fn
    logging.info('convert from genotype: {}'.format(genotype))
    logging.debug('converter: {}'.format(convert_fn.__qualname__))
    Slot._convert_fn = convert_fn
    if hasattr(model, 'build_from_genotype'):
        model.build_from_genotype(genotype, *args, **kwargs)
    else:
        for gene, m in zip(genotype.ops, Slot.slots_all()):
            m.build_from_genotype(gene, *args, **kwargs)
    if drop_path:
        for m in Slot.slots_all():
            if m.fixed: continue
            ent = apply_drop_path(m.ent, drop_path)
            m.set_entity(ent)
    return model

def default_layer_converter(slot, layer_cls, *args, **kwargs):
    if not 'edge_cls' in kwargs:
        kwargs['edge_cls'] = Slot
    if not 'edge_kwargs' in kwargs:
        kwargs['edge_kwargs'] = {
            'chn_in': None,
            'chn_out': None,
            'stride': None,
        }
    ent = build_layer(layer_cls, 
                    chn_in=slot.chn_in, 
                    chn_out=slot.chn_out, 
                    stride=slot.stride, 
                    *args, **kwargs)
    return ent

def convert_from_layers(model, layers_conf, convert_fn=None, *args, **kwargs):
    logging.info('building net layers')
    for i, layer_conf in enumerate(layers_conf):
        layer_convert_fn = default_layer_converter if i >= len(convert_fn) else convert_fn[i]
        if layer_convert_fn is None: layer_convert_fn = default_layer_converter
        layer_cls = layer_conf.type
        layer_args = layer_conf.get('args', {})
        cur_slots = list(Slot.slots_all())
        for m in cur_slots:
            if m.ent is None:
                m.set_entity(layer_convert_fn(m, layer_cls, *args, **kwargs, **layer_args))
                m.fixed = True
        new_slots = [m for m in Slot.slots_all() if m.ent is None]
        logging.debug('new slots from layer: {}'.format(len(new_slots)))