import logging
from functools import partial


class Registry():
    def __init__(self, name='root'):
        self.name = name
        self._reg_class = {}

    def all(self):
        return self._reg_class.values()

    def get_reg_name(self, name):
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass, _reg_id):
        _reg_id = self.get_reg_name(_reg_id)
        if _reg_id in self._reg_class:
            raise ValueError('Cannot re-register _reg_id: {}'.format(_reg_id))
        self._reg_class[_reg_id] = regclass

    def update(self, regdict):
        self._reg_class.update(regdict)

    def get(self, _reg_id):
        _reg_id = self.get_reg_name(_reg_id)
        if _reg_id not in self._reg_class:
            raise ValueError('id \'{}\' not found in registry {}'.format(_reg_id, self.name))
        return self._reg_class[_reg_id]


registry = Registry()


def get_full_path(_reg_path, _reg_id):
    return '{}.{}'.format(_reg_path, _reg_id)


def register(_reg_path, builder, _reg_id=None):
    if _reg_id is None:
        _reg_id = builder.__qualname__
    _reg_id = get_full_path(_reg_path, _reg_id)
    registry.register(builder, _reg_id)
    logging.info('registered: {}'.format(_reg_id))
    return builder


def get_builder(_reg_path, _reg_id):
    return registry.get(get_full_path(_reg_path, _reg_id))


def build(_reg_path, _reg_id, *args, **kwargs):
    return registry.get(get_full_path(_reg_path, _reg_id))(*args, **kwargs)


def register_as(_reg_path, _reg_id=None):
    def reg_builder(func):
        register(_reg_path, func, _reg_id)
        return func

    return reg_builder


def get_registry_utils(_reg_path):
    # _registry = Registry(name)
    _register = partial(register, _reg_path)
    _get_builder = partial(get_builder, _reg_path)
    _build = partial(build, _reg_path)
    _register_as = partial(register_as, _reg_path)
    return _reg_path, _register, _get_builder, _build, _register_as
