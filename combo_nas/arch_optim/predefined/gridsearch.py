import logging
import time
import random
import numpy as np
from ..base import ArchOptimBase
from ...utils import accuracy
from ...core.param_space import ArchParamSpace

class DiscreteSpaceArchOptim(ArchOptimBase):
    def __init__(self, config):
        super().__init__(config)
        self.space_size = ArchParamSpace.discrete_size
        logging.debug('arch space size: {}'.format(self.space_size()))

    def _next(self):
        pass

    def next(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.has_next():
                batch.append(self._next())
        return batch


class GridSearch(DiscreteSpaceArchOptim):
    def __init__(self, config):
        super().__init__(config)
        self.counter = 0
    
    def _next(self):
        index = self.counter
        self.counter = self.counter + 1
        return ArchParamSpace.get_discrete_map(index)
    
    def has_next(self):
        return self.counter < self.space_size()


class RandomSearch(DiscreteSpaceArchOptim):
    def __init__(self, config, seed=None):
        super().__init__(config)
        self.visited = set()
        seed = int(time.time()) if seed is None else seed
        random.seed(seed)
    
    def _next(self):
        index = random.randint(0, self.space_size())
        while index in self.visited:
            index = random.randint(0, self.space_size())
        self.visited.add(index)
        return ArchParamSpace.get_discrete_map(index)
    
    def has_next(self):
        return len(self.visited) < self.space_size()