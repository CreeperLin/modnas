import copy
from ..slot import Slot
from ..mixed_ops import MixedOp
from . import register


@register
class DefaultSlotTraversalExporter():
    """Exporter that outputs parameter values."""

    def __init__(self, export_fn='to_arch_desc', fn_args=None, gen=None):
        self.gen = gen
        self.export_fn = export_fn
        self.fn_args = fn_args or {}
        self.visited = set()

    def export(self, slot, *args, **kwargs):
        """Return exported archdesc from Slot."""
        if slot in self.visited:
            return None
        self.visited.add(slot)
        export_fn = getattr(slot.get_entity(), self.export_fn, None)
        return None if export_fn is None else export_fn(*args, **kwargs)

    def __call__(self, model):
        """Run Exporter."""
        Slot.set_export_fn(self.export)
        arch_desc = []
        gen = self.gen or Slot.gen_slots_model(model)
        for m in gen():
            if m in self.visited:
                continue
            arch_desc.append(m.to_arch_desc(**copy.deepcopy(self.fn_args)))
        self.visited.clear()
        return arch_desc


@register
class DefaultRecursiveExporter():
    """Exporter that recursively outputs archdesc of submodules."""

    def __init__(self, export_fn='to_arch_desc', fn_args=None):
        self.fn_args = fn_args or {}
        self.export_fn = export_fn
        self.visited = set()

    def export(self, slot, *args, **kwargs):
        """Return exported archdesc from Slot."""
        if slot in self.visited:
            return None
        self.visited.add(slot)
        export_fn = getattr(slot.get_entity(), self.export_fn, None)
        return None if export_fn is None else export_fn(*args, **kwargs)

    def visit(self, module):
        """Return exported archdesc from current module."""
        export_fn = getattr(module, self.export_fn, None)
        if export_fn is not None:
            return export_fn(**copy.deepcopy(self.fn_args))
        return {n: self.visit(m) for n, m in module.named_children()}

    def __call__(self, model):
        """Run Exporter."""
        Slot.set_export_fn(self.export)
        desc = self.visit(model)
        self.visited.clear()
        return desc


@register
class DefaultMixedOpExporter():
    """Exporter that outputs archdesc from mixed operators."""

    def __init__(self, fn_args=None):
        self.fn_args = fn_args or {}

    def __call__(self, model):
        """Run Exporter."""
        desc = [m.to_arch_desc(**self.fn_args) for m in MixedOp.gen(model)]
        return desc
