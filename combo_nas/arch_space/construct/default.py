"""Default Constructors."""
import importlib
import logging
import copy
from collections import OrderedDict
from .. import build as build_module
from ..slot import Slot
from . import register, build


def get_convert_fn(convert_fn, **kwargs):
    """Return a new convert function."""
    if isinstance(convert_fn, str):
        convert_fn = build(convert_fn, **kwargs)
    elif callable(convert_fn):
        convert_fn = convert_fn(**kwargs)
    else:
        raise ValueError('unsupported convert_fn type: {}'.format(type(convert_fn)))
    return convert_fn


@register
class DefaultModelConstructor():
    """Constructor that builds model from registered architectures."""

    def __init__(self, model_type, args=None):
        self.model_type = model_type
        self.args = args or {}

    def __call__(self, model):
        """Run constructor."""
        model = build_module(self.model_type, **copy.deepcopy(self.args))
        return model


@register
class ExternalModelConstructor():
    """Constructor that builds model from external sources or libraries."""

    def __init__(self, model_type, src_path=None, import_path=None, args=None):
        self.model_type = model_type
        self.import_path = import_path
        self.src_path = src_path
        self.args = args or {}

    def __call__(self, model):
        """Run constructor."""
        if self.src_path is not None:
            logging.info('Importing model from path: {}'.format(self.src_path))
            name = ''
            spec = importlib.util.spec_from_file_location(name, self.src_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        elif self.import_path is not None:
            logging.info('Importing model from lib: {}'.format(self.import_path))
            mod = importlib.import_module(self.import_path)
        model = mod.__dict__[self.model_type](**self.args)
        return model


@register
class DefaultSlotTraversalConstructor():
    """Constructor that traverses and converts Slots."""

    def __init__(self, gen=None, convert_fn=None, args=None):
        self.gen = gen
        if convert_fn:
            self.convert = get_convert_fn(convert_fn, **(args or {}))

    def convert(self, slot):
        """Return converted module from slot."""
        raise NotImplementedError

    def __call__(self, model):
        """Run constructor."""
        gen = self.gen or Slot.gen_slots_model(model)
        all_slots = list(gen())
        for m in all_slots:
            if m.ent is not None:
                continue
            ent = self.convert(m)
            m.set_entity(ent)
        return model


@register
class DefaultMixedOpConstructor(DefaultSlotTraversalConstructor):
    """Default Mixed Operator Constructor."""

    def __init__(self, primitives, mixed_type, mixed_args=None, primitive_args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        self.primitives = primitives
        self.mixed_type = mixed_type
        self.mixed_args = mixed_args or {}
        self.primitive_args = primitive_args or {}

    def convert(self, slot):
        """Return converted MixedOp from slot."""
        prims = OrderedDict([(prim, build_module(prim, slot, **self.primitive_args)) for prim in self.primitives])
        return build_module(self.mixed_type, primitives=prims, **self.mixed_args)


@register
class DefaultOpConstructor(DefaultSlotTraversalConstructor):
    """Default Network Operator Constructor."""

    def __init__(self, op_type, args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        self.args = args or {}
        self.op_type = op_type

    def convert(self, slot):
        """Return converted operator from slot."""
        return build_module(self.op_type, slot, **copy.deepcopy(self.args))
