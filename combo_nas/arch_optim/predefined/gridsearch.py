import logging
import time
import numpy as np
from ..base import ArchOptimBase
from ...utils import accuracy
from ...core.param_space import ArchParamSpace

class DiscreteSpaceArchOptim(ArchOptimBase):
    def __init__(self, config, net):
        super().__init__(config, net)
        self.space_size = ArchParamSpace.discrete_size()
        logging.debug('arch space size: {}'.format(self.space_size))

class GridSearch(DiscreteSpaceArchOptim):
    def __init__(self, config, net):
        super().__init__(config, net)
        self.counter = 0
    
    def step(self, estim):
        if self.counter >= self.space_size: return None
        index = self.counter
        self.counter = self.counter + 1
        return ArchParamSpace.get_discrete(index)


class RandomSearch(DiscreteSpaceArchOptim):
    def __init__(self, config, net, seed=None):
        super().__init__(config, net)
        self.visited = set()
        seed = int(time.time()) if seed is None else seed
        rng = np.random.RandomState()
        rng.seed(seed)
        self.np_random = rng
    
    def step(self, estim):
        if len(self.visited) >= self.space_size: return None
        index = self.np_random.randint(self.space_size)
        while index in self.visited:
            index = self.np_random.randint(self.space_size)
        self.visited.add(index)
        ArchParamSpace.set_discrete(index)