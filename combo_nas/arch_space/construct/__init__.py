from ...utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('construct')

from . import default, arch_desc, model_init, torch