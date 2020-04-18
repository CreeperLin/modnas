from .gradient_based import DARTSOptim, BinaryGateOptim, DirectGradOptim,\
                            DirectGradBiLevelOptim, REINFORCEOptim,\
                            GumbelAnnealingOptim
from .gridsearch import RandomSearchOptim, GridSearchOptim
from .model_based import ModelBasedOptim
from .genetic import EvolutionOptim, RegularizedEvolutionOptim

from .. import register
register(DARTSOptim, 'DARTS')
register(BinaryGateOptim, 'BinGate')
register(DirectGradOptim, 'DirectGrad')
register(DirectGradBiLevelOptim, 'DirectGradBiLevel')
register(REINFORCEOptim, 'REINFORCE')
register(GumbelAnnealingOptim, 'GumbelAnnealing')
register(RandomSearchOptim, 'Random')
register(GridSearchOptim, 'GridSearch')
register(ModelBasedOptim, 'ModelBased')
register(EvolutionOptim, 'Evolution')
register(RegularizedEvolutionOptim, 'RegularizedEvolution')
