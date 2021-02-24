"""Registry for framework components."""
import sys
import logging
import traceback
from functools import partial


class Registry():
    """Registry class."""

    def __init__(self, name='root', allow_replace=True):
        self.name = name
        self.allow_replace = allow_replace
        self._reg_class = {}

    def get_reg_name(self, name):
        """Return proper registration name."""
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass, _reg_id):
        """Register a component class."""
        _reg_id = self.get_reg_name(_reg_id)
        if _reg_id in self._reg_class and not self.allow_replace:
            raise ValueError('Cannot re-register _reg_id: {}'.format(_reg_id))
        self._reg_class[_reg_id] = regclass

    def update(self, regdict):
        """Update registry."""
        self._reg_class.update(regdict)

    def get(self, _reg_id):
        """Return registered class by name."""
        _reg_id = self.get_reg_name(_reg_id)
        if _reg_id not in self._reg_class:
            raise ValueError('id \'{}\' not found in registry'.format(_reg_id))
        return self._reg_class[_reg_id]


registry = Registry()


def get_full_path(_reg_path, _reg_id):
    """Return full registration path."""
    return '{}.{}'.format(_reg_path, _reg_id)


def register(_reg_path, builder, _reg_id=None):
    """Register class as name."""
    if _reg_id is None:
        _reg_id = builder.__qualname__
    _reg_id = get_full_path(_reg_path, _reg_id)
    registry.register(builder, _reg_id)
    return builder


def get_builder(_reg_path, _reg_id):
    """Return class builder by name."""
    return registry.get(get_full_path(_reg_path, _reg_id))


def parse_spec(spec):
    if isinstance(spec, dict):
        return spec.get('type'), spec.get('args', {})
    if isinstance(spec, (tuple, list)) and isinstance(spec[0], str):
        return spec[0], {} if len(spec) < 2 else spec[1]
    if isinstance(spec, str):
        return spec, {}
    raise ValueError('Invalid build spec: {}'.format(spec))


def to_spec(reg_id, kwargs):
    return {
        'type': reg_id,
        'args': kwargs
    }


def build(_reg_path, _spec, *args, **kwargs):
    """Instantiate class by name."""
    reg_id, sp_kwargs = parse_spec(_spec)
    kwargs.update(sp_kwargs)
    return registry.get(get_full_path(_reg_path, reg_id))(*args, **kwargs)


def register_as(_reg_path, _reg_id=None):
    """Return a registration decorator."""
    def reg_builder(func):
        register(_reg_path, func, _reg_id)
        return func

    return reg_builder


def get_registry_utils(_reg_path):
    """Return registration utilities."""
    _register = partial(register, _reg_path)
    _get_builder = partial(get_builder, _reg_path)
    _build = partial(build, _reg_path)
    _register_as = partial(register_as, _reg_path)
    return _reg_path, _register, _get_builder, _build, _register_as


def get_registry_name(path):
    return '.'.join(path[path.index('modnas') + 2:])


class RegistryModule():
    """Registry as a module."""

    def __init__(self, fullname):
        path = fullname.split('.')
        registry_name = get_registry_name(path)
        self.__package__ = fullname
        self.__path__ = path
        self.__name__ = registry_name
        self.__loader__ = None
        self.__spec__ = None
        self.reg_path, self.register, self.get_builder, self.build, self.register_as = get_registry_utils(registry_name)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__.get(attr)
        return self.get_builder(attr)


class RegistryImporter():
    """Create new Registry using import hooks (PEP 302)."""

    def find_module(self, fullname, path=None):
        """Handle registry imports."""
        if 'modnas.registry' in fullname:
            return self

    def load_module(self, fullname):
        """Create and find registry by import path."""
        path = fullname.split('.')
        reg_path, reg_id = path[:-1], path[-1]
        reg_fullname = '.'.join(reg_path)
        registry_name = get_registry_name(reg_path)
        if reg_fullname in sys.modules and len(registry_name):
            mod = get_builder(registry_name, reg_id)
            sys.modules[fullname] = mod
            return mod
        mod = sys.modules.get(fullname, RegistryModule(fullname))
        sys.modules[fullname] = mod
        return mod


sys.meta_path.append(RegistryImporter())
