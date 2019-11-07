import logging
import torch
import torch.nn as nn
from ..core.ops import build_op, Identity, DropPath_
from ..core.mixed_ops import build_mixed_op
from . import genotypes as gt

class Slot(nn.Module):
    _slot_id = -1
    _param_id = -1
    _convert_fn = None

    def __init__(self, chn_in, chn_out, stride, name=None, pid=None, *args, **kwargs):
        super().__init__()
        self.sid = Slot.new_slot_id()
        self.pid = Slot.new_param_id() if pid is None else pid
        self.param_pid = pid
        self.name = str(self.sid) if name is None else name
        self.chn_in = chn_in
        chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_in if chn_out is None else chn_out
        self.stride = stride
        self.ent = None
        self.gene = None
        self.args = args
        self.kwargs = kwargs
        logging.debug('slot {} ({} {}) {}: declared {} {} {}'.format(
            self.sid, self.param_pid, self.pid, self.name, self.chn_in, self.chn_out, self.stride))
    
    @staticmethod
    def reset():
        Slot._slot_id = -1
        Slot._param_id = -1
        Slot._convert_fn = None

    @staticmethod
    def new_slot_id():
        Slot._slot_id += 1
        return Slot._slot_id

    @staticmethod
    def new_param_id():
        Slot._param_id += 1
        return Slot._param_id
    
    def set_entity(self, ent):
        self.ent = ent
        logging.debug('slot {} ({}) {}: set to {}'.format(self.sid, self.pid, self.name, ent.__class__.__name__))
    
    def forward(self, x):
        if self.ent is None:
            raise ValueError('Undefined entity in slot {}'.format(self.sid))
        x = x[0] if isinstance(x, list) else x
        return self.ent(x)

    def to_genotype(self, *args, **kwargs):
        try:
            return self.ent.to_genotype(*args, **kwargs)
        except:
            logging.debug('slot {} default genotype {}'.format(self.sid, self.gene))
            return 1, self.gene

    def build_from_genotype(self, gene, *args, **kwargs):
        convert_fn = default_genotype_converter if Slot._convert_fn is None else Slot._convert_fn
        ent = convert_fn(self, gene, *args, **kwargs)
        self.gene = gene
        self.set_entity(ent)


def slots(model):
    for n, m in model.named_modules():
        if isinstance(m, Slot):
            yield m

def default_predefined_converter(slot, mixed_op_cls, *args, **kwargs):
    ent = build_mixed_op(mixed_op_cls, 
                        chn_in=slot.chn_in, 
                        chn_out=slot.chn_out, 
                        stride=slot.stride, 
                        ops=gt.get_primitives(), 
                        pid=slot.param_pid, 
                        *args, **kwargs)
    return ent

def default_op_builder(op_name, slot, drop_path=True):
    chn_in = slot.chn_in if isinstance(slot.chn_in, int) else slot.chn_in[0]
    chn_out = slot.chn_out if isinstance(slot.chn_out, int) else slot.chn_out[0]
    ent = build_op(op_name, chn_in, chn_out, slot.stride)
    if drop_path and not isinstance(ent, Identity):
        ent = nn.Sequential(
            ent,
            DropPath_()
        )
    return ent

def default_genotype_converter(slot, gene, drop_path=True):
    if isinstance(gene, list): gene = gene[0]
    op_name = gene
    ent = default_op_builder(op_name, slot, drop_path)
    return ent

def convert_from_predefined_net(model, convert_fn=None, *args, **kwargs):
    convert_fn = default_predefined_converter if convert_fn is None else convert_fn
    for m in slots(model):
        ent = convert_fn(m, *args, **kwargs)
        m.set_entity(ent)
    return model

def convert_from_genotype(model, genotype, convert_fn=None, *args, **kwargs):
    convert_fn = default_genotype_converter if convert_fn is None else convert_fn
    logging.info('building net from genotype: {}'.format(genotype))
    try:
        Slot._convert_fn = convert_fn
        model.build_from_genotype(genotype, *args, **kwargs)
    except AttributeError:
        for i, m in enumerate(slots(model)):
            gene = genotype.ops[i]
            ent = convert_fn(m, gene, *args, **kwargs)
            m.set_entity(ent)
    return model
