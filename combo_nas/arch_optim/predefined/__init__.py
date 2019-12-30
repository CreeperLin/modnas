from .gradient_based import DARTSArchOptim, BinaryGateArchOptim, DirectGradArchOptim, REINFORCEArchOptim
from .gridsearch import RandomSearchArchOptim, GridSearchArchOptim
from ...utils.registration import Registry, build, get_builder, register, register_wrapper
from functools import partial

arch_optim_registry = Registry('arch_optim')
register_arch_optim = partial(register, arch_optim_registry)
get_arch_optim_builder = partial(get_builder, arch_optim_registry)
build_arch_optim = partial(build, arch_optim_registry)
register = partial(register_wrapper, arch_optim_registry)

register_arch_optim(DARTSArchOptim, 'WeightedSum')
register_arch_optim(BinaryGateArchOptim, 'BinGate')
register_arch_optim(DirectGradArchOptim, 'Dummy')
register_arch_optim(REINFORCEArchOptim, 'REINFORCE')
register_arch_optim(RandomSearchArchOptim, 'Random')
register_arch_optim(GridSearchArchOptim, 'GridSearch')