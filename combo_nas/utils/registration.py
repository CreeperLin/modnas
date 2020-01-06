import importlib
import logging

class Registry(object):
    """
    registry
    """

    def __init__(self, name=''):
        self.name = name
        self._reg_class = {}

    def make(self, path):
        try:
            module = importlib.import_module(path)
            # catch ImportError for python2.7 compatibility
        except ImportError:
            raise ValueError('A module ({}) not found'.format(path))
        return module

    def all(self):
        return self._reg_class.values()

    def register(self, regclass, rid=None):
        rid = regclass.__name__ if rid is None else rid
        rid = rid.lower()
        if rid in self._reg_class:
            raise ValueError('Cannot re-register rid: {}'.format(rid))
        self._reg_class[rid] = regclass

    def update(self, regdict):
        self._reg_class.update(regdict)

    def get(self, rid):
        rid = rid.lower()
        if not rid in self._reg_class: raise ValueError('registry id not found: {}'.format(rid))
        return self._reg_class[rid]


def register(reg, net_builder, rid=None):
    reg.register(net_builder, rid)
    logging.info('registered {}: {}'.format(reg.name, rid))

def get_builder(reg, rid):
    return reg.get(rid)

def build(reg, rid, *args, **kwargs):
    # logging.debug('build {}: {}'.format(reg.name, rid))
    return reg.get(rid)(*args, **kwargs)

def register_wrapper(reg, rid):
    def reg_builder(func):
        register(reg, func, rid)
        def reg_builder_rid(*args, **kwargs):
            return func(*args, **kwargs)
        return reg_builder_rid
    return reg_builder