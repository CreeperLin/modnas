# -*- coding: utf-8 -*-
from functools import partial
from ..utils.registration import Registry, build, get_builder, register, register_as

arch_space_registry = Registry('arch_space')
register_arch_space = partial(register, arch_space_registry)
get_arch_space_builder = partial(get_builder, arch_space_registry)
build_arch_space = partial(build, arch_space_registry)
register = partial(register_as, arch_space_registry)

from . import predefined