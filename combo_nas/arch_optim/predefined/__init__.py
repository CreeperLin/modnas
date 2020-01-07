from .gradient_based import DARTSArchOptim, BinaryGateArchOptim, DirectGradArchOptim, REINFORCEArchOptim
from .gridsearch import RandomSearchArchOptim, GridSearchArchOptim

from .. import register_arch_optim
register_arch_optim(DARTSArchOptim, 'WeightedSum')
register_arch_optim(BinaryGateArchOptim, 'BinGate')
register_arch_optim(DirectGradArchOptim, 'DirectGrad')
register_arch_optim(REINFORCEArchOptim, 'REINFORCE')
register_arch_optim(RandomSearchArchOptim, 'Random')
register_arch_optim(GridSearchArchOptim, 'GridSearch')