from functools import partial
from ..utils.registration import Registry, build, get_builder, register, register_wrapper

estimator_registry = Registry('estimator')
register_estimator = partial(register, estimator_registry)
get_estimator_builder = partial(get_builder, estimator_registry)
build_estimator = partial(build, estimator_registry)
register = partial(register_wrapper, estimator_registry)

from . import predefined