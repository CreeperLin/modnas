import copy
import logging
from collections import OrderedDict
import torch.nn as nn
from . import ops, mixed_ops, layers
from . import build

class Slot(nn.Module):
    _slots = []
    _slot_id = -1
    _convert_fn = None
    _visited = set()

    def __init__(self, chn_in, chn_out, stride, name=None, kwargs=None):
        super().__init__()
        Slot.register(self)
        self.name = str(self.sid) if name is None else name
        self.e_chn_in = chn_in
        self.e_chn_out = chn_in if chn_out is None else chn_out
        self.stride = stride
        self.ent = None
        self.gene = None
        self.kwargs = {} if kwargs is None else kwargs
        self.fixed = False
        logging.debug('slot {} {}: declared {} {} {}'.format(
            self.sid, self.name, self.chn_in, self.chn_out, self.stride))

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
        return chn_in[0] if isinstance(chn_in, (list, tuple)) and len(chn_in) == 1 else chn_in

    @property
    def chn_out(self):
        chn_out = self.e_chn_out
        return chn_out[0] if isinstance(chn_out, (list, tuple)) and len(chn_out) == 1 else chn_out

    @staticmethod
    def gen_slots_all():
        for m in Slot._slots:
            yield m


    @staticmethod
    def gen_slots_model(model):
        def gen():
            for m in model.modules():
                if isinstance(m, Slot):
                    yield m
        return gen

    @staticmethod
    def call_all(funcname, gen=None, fn_kwargs=None):
        if gen is None: gen = Slot.gen_slots_all
        ret = []
        for m in gen():
            if hasattr(m, funcname):
                ret.append(getattr(m, funcname)(**({} if fn_kwargs is None else copy.deepcopy(fn_kwargs))))
        return ret

    @staticmethod
    def apply_all(func, gen=None, fn_kwargs=None):
        if gen is None: gen = Slot.gen_slots_all
        ret = []
        for m in gen():
            ret.append(func(m, **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs))))
        return ret

    @staticmethod
    def reset_visited():
        Slot._visited = set()

    @staticmethod
    def set_visited(slot):
        if not Slot._visited is None:
            Slot._visited.add(slot)

    @staticmethod
    def is_visited(slot):
        return not Slot._visited is None and slot in Slot._visited

    @staticmethod
    def to_genotype_all(gen=None, fn_kwargs=None):
        if gen is None: gen = Slot.gen_slots_all
        Slot.reset_visited()
        gene = []
        for m in gen():
            if Slot.is_visited(m):
                continue
            g = m.to_genotype(**({} if fn_kwargs is None else copy.deepcopy(fn_kwargs)))
            gene.append(g)
        return gene

    @staticmethod
    def set_convert_fn(func):
        Slot._convert_fn = func

    def get_entity(self):
        return self._modules.get('ent', None)

    def set_entity(self, ent):
        if self.fixed:
            return
        self.ent = ent
        logging.debug('slot {} {}: set to {}'.format(self.sid, self.name, ent.__class__.__name__))

    def del_entity(self):
        if self.fixed:
            return
        if self.ent is None:
            return
        del self.ent

    def forward(self, *args, **kwargs):
        if self.ent is None:
            raise ValueError('Undefined entity in slot {}'.format(self.sid))
        return self.ent(*args, **kwargs)

    def to_genotype(self, *args, **kwargs):
        Slot.set_visited(self)
        if hasattr(self.ent, 'to_genotype'):
            return self.ent.to_genotype(*args, **kwargs)
        else:
            logging.debug('slot {} default genotype {}'.format(self.sid, self.gene))
            return self.gene

    def build_from_genotype(self, gene, *args, **kwargs):
        self.gene = gene
        if self.ent is None:
            convert_fn = default_genotype_converter if Slot._convert_fn is None else Slot._convert_fn
            ent = convert_fn(self, gene, *args, **kwargs)
            self.set_entity(ent)
        elif not Slot.is_visited(self):
            self.ent.build_from_genotype(gene, *args, **kwargs)
        Slot.set_visited(self)

    def extra_repr(self):
        expr = '{}, {}, {}, '.format(self.chn_in, self.chn_out, self.stride)+\
                ', '.join(['{}={}'.format(k, v) for k, v in self.kwargs.items()])
        return expr


def default_mixed_op_converter(slot, primitives, mixed_op_type, mixed_op_args=None, primitive_args=None):
    if mixed_op_args is None:
        mixed_op_args = {}
    if primitive_args is None:
        primitive_args = {}
    primitives = OrderedDict([
        (prim, ops.build(prim, slot.chn_in, slot.chn_out, slot.stride, **primitive_args)) for prim in primitives
    ])
    ent = mixed_ops.build(mixed_op_type,
                          primitives=primitives,
                          **mixed_op_args)
    return ent


def default_genotype_converter(slot, gene, op_args=None):
    if isinstance(gene, list): gene = gene[0]
    op_name = gene
    ent = ops.build(op_name, slot.chn_in, slot.chn_out, slot.stride, **({} if op_args is None else copy.deepcopy(op_args)))
    return ent


def convert_from_predefined_net(model, convert_fn, gen=None, fn_kwargs=None):
    """Convert Slots to actual modules using predefined converter function only.

    """
    if gen is None: gen = Slot.gen_slots_all
    logging.info('convert from predefined net')
    for m in gen():
        if m.fixed: continue
        ent = convert_fn(m, **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs)))
        m.set_entity(ent)
    return model


def convert_from_genotype(model, genotype, convert_fn=None, gen=None, fn_kwargs=None):
    """Convert Slots to actual modules from genotype.

    """
    if gen is None: gen = Slot.gen_slots_all
    convert_fn = default_genotype_converter if convert_fn is None else convert_fn
    logging.info('convert from genotype: {}'.format(genotype))
    Slot.set_convert_fn(convert_fn)
    if hasattr(model, 'build_from_genotype'):
        model.build_from_genotype(genotype, **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs)))
    else:
        for gene, m in zip(genotype.ops, gen()):
            m.build_from_genotype(gene, **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs)))
    return model


def default_layer_converter(slot, layer_cls, layer_kwargs=None):
    layer_kwargs = {} if layer_kwargs is None else layer_kwargs
    if not 'edge_cls' in layer_kwargs:
        layer_kwargs['edge_cls'] = Slot
    if not 'edge_kwargs' in layer_kwargs:
        layer_kwargs['edge_kwargs'] = {
            'chn_in': None,
            'chn_out': None,
            'stride': None,
        }
    ent = layers.build(layer_cls,
                       chn_in=slot.chn_in,
                       chn_out=slot.chn_out,
                       stride=slot.stride,
                       **layer_kwargs)
    return ent


def convert_from_layers(model, layers_conf, convert_fn=None, gen=None, fn_kwargs=None):
    """Convert Slots to predefined layers using series of converter function.

    """
    del model
    if gen is None: gen = Slot.gen_slots_all
    logging.info('building net layers')
    for i, layer_conf in enumerate(layers_conf):
        layer_convert_fn = default_layer_converter if i >= len(convert_fn) else convert_fn[i]
        if layer_convert_fn is None: layer_convert_fn = default_layer_converter
        layer_cls = layer_conf.type
        layer_args = layer_conf.get('args', {})
        cur_slots = list(gen())
        for m in cur_slots:
            if m.ent is None:
                layer = layer_convert_fn(m, layer_cls,
                                         layer_kwargs=copy.deepcopy(layer_args),
                                         **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs)))
                m.set_entity(layer)
                m.fixed = True
        new_slots = [m for m in gen() if m.ent is None]
        logging.debug('new slots from layer: {}'.format(len(new_slots)))


def build_predefined(*args, **kwargs):
    net = build(*args, **kwargs)
    try:
        convert_fn = net.get_predefined_augment_converter()
    except AttributeError:
        convert_fn = None
    net = convert_from_predefined_net(net, convert_fn=convert_fn)
    return net
