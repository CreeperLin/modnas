from .gradient_based import DARTSOptim, BinaryGateOptim, DirectGradOptim, REINFORCEOptim
from .gridsearch import RandomSearchOptim, GridSearchOptim
from .model_based import ModelBasedOptim

from .. import register
register(DARTSOptim, 'WeightedSum')
register(BinaryGateOptim, 'BinGate')
register(DirectGradOptim, 'DirectGrad')
register(REINFORCEOptim, 'REINFORCE')
register(RandomSearchOptim, 'Random')
register(GridSearchOptim, 'GridSearch')
register(ModelBasedOptim, 'ModelBased')
