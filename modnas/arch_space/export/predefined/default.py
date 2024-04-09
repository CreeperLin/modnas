"""Default Architecture Exporters."""
import os
import json
import yaml
from modnas.core.param_space import ParamSpace
from modnas.registry.export import register, build
from typing import Any, Dict, List, Optional, Union


@register
class DefaultToFileExporter():
    """Exporter that saves archdesc to file."""

    def __init__(self, path: str, ext: str = 'yaml') -> None:
        path, pathext = os.path.splitext(path)
        ext = pathext or ext
        path = path + '.' + ext
        self.path = path
        self.ext = ext

    def __call__(self, desc: Any) -> None:
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
        self.exporters = {k: build(exp['type'], **exp.get('args', {})) for k, exp in exporters.items()}

    def __call__(self, model):
        """Run Exporter."""
        return {k: exp(model) for k, exp in self.exporters.items()}


@register
class DefaultParamsExporter():
    """Exporter that outputs parameter values."""

    def __init__(self, export_fmt: Optional[str] = None, with_keys: bool = True) -> None:
        self.export_fmt = export_fmt
        self.with_keys = with_keys

    def __call__(self, model: None) -> Union[Dict[str, Any], List[Any], str]:
        """Run Exporter."""
        params = dict(ParamSpace().named_param_values()) if model is None else model
        if self.with_keys:
            params_dct = params
            return self.export_fmt.format(**params_dct) if self.export_fmt else params_dct
        else:
            params_list = list(params.values())
            return self.export_fmt.format(*params_list) if self.export_fmt else params_list
