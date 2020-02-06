# -*- coding: utf-8 -*-
from functools import partial
from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('arch_space')

from . import predefined