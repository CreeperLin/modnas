# -*- coding: utf-8 -*-
from functools import partial
from ..utils.registration import Registry, build, get_builder, register, register_as

optim_registry = Registry('optim')
register_optim = partial(register, optim_registry)
get_optim_builder = partial(get_builder, optim_registry)
build_optim = partial(build, optim_registry)
register = partial(register_as, optim_registry)

from . import predefined
