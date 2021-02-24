"""Default Constructors."""
import importlib
import logging
import copy
from collections import OrderedDict
from modnas.registry.arch_space import build as build_module
from modnas.registry.construct import register, build
from modnas.arch_space.slot import Slot


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

    def __init__(self, gen=None, convert_fn=None, args=None, skip_exist=True):
        self.gen = gen
        self.skip_exist = skip_exist
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
            if self.skip_exist and m.get_entity() is not None:
                continue
            ent = self.convert(m)
            if ent is not None:
                m.set_entity(ent)
        return model


@register
class DefaultMixedOpConstructor(DefaultSlotTraversalConstructor):
    """Default Mixed Operator Constructor."""

    def __init__(self, primitives, mixed_op, primitive_args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        self.primitives = primitives
        self.mixed_op_conf = mixed_op
        self.primitive_args = primitive_args or {}

    def convert(self, slot):
        """Return converted MixedOp from slot."""
        prims = OrderedDict([(prim, build_module(prim, slot, **self.primitive_args)) for prim in self.primitives])
        return build_module(self.mixed_op_conf, primitives=prims)


@register
class DefaultOpConstructor(DefaultSlotTraversalConstructor):
    """Default Network Operator Constructor."""

    def __init__(self, op):
        DefaultSlotTraversalConstructor.__init__(self)
        self.op_conf = op

    def convert(self, slot):
        """Return converted operator from slot."""
        return build_module(copy.deepcopy(self.op_conf), slot)
