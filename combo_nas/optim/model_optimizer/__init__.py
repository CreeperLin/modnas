from ...utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('model_optimizer')

from .sampling import RandomSamplingModelOptimizer
from .sa import SimulatedAnnealingModelOptimizer

register(RandomSamplingModelOptimizer, 'RandomSampling')
register(SimulatedAnnealingModelOptimizer, 'SimulatedAnnealing')