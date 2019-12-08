from .default_estimator import DefaultEstimator
from .supernet_estimator import SuperNetEstimator
from .subnet_estimator import SubNetEstimator
from ...utils.registration import Registry, build, get_builder, register, register_wrapper
from functools import partial

estimator_registry = Registry('estimator')
register_estimator = partial(register, estimator_registry)
get_estimator_builder = partial(get_builder, estimator_registry)
build_estimator = partial(build, estimator_registry)
register = partial(register_wrapper, estimator_registry)

register_estimator(DefaultEstimator, 'Default')
register_estimator(SuperNetEstimator, 'SuperNet')
register_estimator(SubNetEstimator, 'SubNet')