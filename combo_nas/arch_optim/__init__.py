# -*- coding: utf-8 -*-
from functools import partial
from ..utils.registration import Registry, build, get_builder, register, register_wrapper

arch_optim_registry = Registry('arch_optim')
register_arch_optim = partial(register, arch_optim_registry)
get_arch_optim_builder = partial(get_builder, arch_optim_registry)
build_arch_optim = partial(build, arch_optim_registry)
register = partial(register_wrapper, arch_optim_registry)

from . import predefined
