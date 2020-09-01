import os
import yaml
import json
import logging
import copy
from .default import DefaultSlotTraversalConstructor
from ..slot import Slot
from .. import build as build_module
from . import register

_arch_desc_parser = {
    'json': lambda desc: json.loads(desc),
    'yaml': lambda desc: yaml.load(desc, Loader=yaml.FullLoader),
    'yml': lambda desc: yaml.load(desc, Loader=yaml.FullLoader),
}


class DefaultArchDescConstructor():
    def __init__(self, arch_desc, parse_args=None):
        arch_desc = self.parse_arch_desc(arch_desc, **(parse_args or {}))
        logging.info('construct from arch_desc: {}'.format(arch_desc))
        self.arch_desc = arch_desc

    def parse_arch_desc(self, desc, parser=None):
        if isinstance(desc, str):
            default_parser = 'yaml'
            if os.path.exists(desc):
                _, ext = os.path.splitext(desc)
                default_parser = ext[1:].lower()
                with open(desc, 'r', encoding='UTF-8') as f:
                    desc = f.read()
            parser = parser or default_parser
            parse_fn = _arch_desc_parser.get(parser)
            if parse_fn is None:
                raise ValueError('invalid arch_desc parser type: {}'.format(parser))
            return parse_fn(desc)
        else:
            return desc

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@register
class DefaultRecursiveArchDescConstructor(DefaultArchDescConstructor):
    def __init__(self, arch_desc, parse_args=None, construct_fn='build_from_arch_desc', fn_args=None, substitute=False):
        super().__init__(arch_desc, parse_args)
        self.construct_fn = construct_fn
        self.fn_args = fn_args or {}
        self.substitute = substitute

    def visit(self, module):
        construct_fn = getattr(module, self.construct_fn, None)
        if construct_fn is not None:
            ret = construct_fn(self.arch_desc, **copy.deepcopy(self.fn_args))
            return module if ret is None else ret
        for n, m in module.named_children():
            m = self.visit(m)
            if m is not None and self.substitute:
                module.add_module(n, m)
        return module

    def __call__(self, model):
        Slot.set_convert_fn(self.convert)
        return self.visit(model)

    def convert(self, slot, desc, *args, **kwargs):
        desc = desc[0] if isinstance(desc, list) else desc
        ent = build_module(desc, slot, *args, **kwargs)
        return ent


@register
class DefaultSlotArchDescConstructor(DefaultSlotTraversalConstructor, DefaultArchDescConstructor):
    def __init__(self, arch_desc, parse_args=None, fn_args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        DefaultArchDescConstructor.__init__(self, arch_desc, parse_args)
        self.fn_args = fn_args or {}
        self.idx = -1

    def get_next_desc(self):
        self.idx += 1
        desc = self.arch_desc[self.idx]
        if isinstance(desc, list) and len(desc) == 1:
            desc = desc[0]
        return desc

    def convert(self, slot):
        m_type = self.get_next_desc()
        return build_module(m_type, slot, **copy.deepcopy(self.fn_args))
