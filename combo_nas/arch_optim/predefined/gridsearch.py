import logging
import time
import random
import numpy as np
from ..base import ArchOptimBase
from ...utils import accuracy

class DiscreteSpaceArchOptim(ArchOptimBase):
    def __init__(self, space):
        super().__init__(space)
        self.space_size = self.space.discrete_size
        logging.debug('arch space size: {}'.format(self.space_size()))

    def _next(self):
        pass

    def next(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.has_next():
                batch.append(self._next())
        return batch


class GridSearchArchOptim(DiscreteSpaceArchOptim):
    def __init__(self, space):
        super().__init__(space)
        self.counter = 0

    def _next(self):
        index = self.counter
        self.counter = self.counter + 1
        return self.space.get_discrete_map(index)

    def has_next(self):
        return self.counter < self.space_size()


class RandomSearchArchOptim(DiscreteSpaceArchOptim):
    def __init__(self, space, seed=None):
        super().__init__(space)
        self.visited = set()
        seed = int(time.time()) if seed is None else seed
        random.seed(seed)

    def _next(self):
        index = random.randint(0, self.space_size())
        while index in self.visited:
            index = random.randint(0, self.space_size())
        self.visited.add(index)
        return self.space.get_discrete_map(index)

    def has_next(self):
        return len(self.visited) < self.space_size()