from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('trainer')

from . import predefined