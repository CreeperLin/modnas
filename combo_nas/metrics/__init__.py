from functools import partial
from ..utils.registration import Registry, build, get_builder, register, register_as

metrics_registry = Registry('metrics')
register_metrics = partial(register, metrics_registry)
get_metrics_builder = partial(get_builder, metrics_registry)
build_metrics = partial(build, metrics_registry)
register = partial(register_as, metrics_registry)

from . import predefined