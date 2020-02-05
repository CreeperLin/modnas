from .gradient_based import DARTSOptim, BinaryGateOptim, DirectGradOptim, REINFORCEOptim
from .gridsearch import RandomSearchOptim, GridSearchOptim
from .model_based import ModelBasedOptim

from .. import register_optim
register_optim(DARTSOptim, 'WeightedSum')
register_optim(BinaryGateOptim, 'BinGate')
register_optim(DirectGradOptim, 'DirectGrad')
register_optim(REINFORCEOptim, 'REINFORCE')
register_optim(RandomSearchOptim, 'Random')
register_optim(GridSearchOptim, 'GridSearch')
register_optim(ModelBasedOptim, 'ModelBased')
