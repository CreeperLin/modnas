"""Default Architecture Exporters."""
import os
import copy
import json
import yaml
from ..slot import Slot
from ..mixed_ops import MixedOp
from ...core.param_space import ParamSpace
from . import register, build


@register
class DefaultToFileExporter():
    """Exporter that saves archdesc to file."""

    def __init__(self, path, ext='yaml'):
        path, pathext = os.path.splitext(path)
        ext = pathext or ext
        path = path + '.' + ext
        self.path = path
        self.ext = ext

    def __call__(self, desc):
        """Run Exporter."""
        ext = self.ext
        if isinstance(desc, str):
            desc_str = desc
        elif ext == 'json':
            desc_str = yaml.dump(desc)
        elif ext in ['yaml', 'yml']:
            desc_str = json.dumps(desc)
        else:
            raise ValueError('invalid arch_desc extension')
        with open(self.path, 'w', encoding='UTF-8') as f:
            f.write(desc_str)


@register
class MergeExporter():
    """Exporter that merges outputs of multiple Exporters."""

    def __init__(self, exporters):
        self.exporters = {k: build(exp_conf) for k, exp_conf in exporters.items()}

    def __call__(self, model):
        """Run Exporter."""
        return {k: exp(model) for k, exp in self.exporters.items()}


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
class DefaultParamsExporter():
    """Exporter that outputs parameter values."""

    def __init__(self, export_fmt=None, with_keys=True):
        self.export_fmt = export_fmt
        self.with_keys = with_keys

    def __call__(self, model):
        """Run Exporter."""
        if self.with_keys:
            params = {k: p.value() for k, p in ParamSpace().named_params()}
        else:
            params = [p.value() for p in ParamSpace().params()]
        if self.export_fmt:
            if self.with_keys:
                return self.export_fmt.format(**params)
            return self.export_fmt.format(*params)
        return params


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
class DefaultMixedOpExporter():
    """Exporter that outputs archdesc from mixed operators."""

    def __init__(self, fn_args=None):
        self.fn_args = fn_args or {}

    def __call__(self, model):
        """Run Exporter."""
        desc = [m.to_arch_desc(**self.fn_args) for m in MixedOp.gen(model)]
        return desc
