import time
import numpy as np
from .tuner import Tuner

class GridSearchTuner(Tuner):
    """Enumerate the search space in a grid search order"""
    def __init__(self, space):
        super(GridSearchTuner, self).__init__(space)
        self.counter = 0

    def next(self):
        if self.counter >= len(self.space): return None
        index = self.counter
        self.counter = self.counter + 1
        return self.space.get(index)

    def has_next(self):
        return self.counter < len(self.space)

    def update(self, inputs, result):
        pass

    def load_history(self, data_set):
        pass

    def __getstate__(self):
        return {"counter": self.counter}

    def __setstate__(self, state):
        self.counter = state['counter']


class RandomTuner(Tuner):
    """Enumerate the search space in a random order"""
    def __init__(self, space, seed=None):
        super(RandomTuner, self).__init__(space)
        self.visited = set()
        seed = int(time.time()) if seed is None else seed
        rng = np.random.RandomState()
        rng.seed(seed)
        self.np_random = rng

    def next(self):
        ret = []
        counter = 0
        if len(self.visited) >= len(self.space): return None
        index = self.np_random.randint(len(self.space))
        while index in self.visited:
            index = self.np_random.randint(len(self.space))
        self.visited.add(index)
        counter += 1
        return self.space.get(index)
    
    def update(self, inputs, result):
        pass

    def has_next(self):
        return len(self.visited) < len(self.space)

    def load_history(self, data_set):
        pass

    def __getstate__(self):
        return {"visited": self.counter}

    def __setstate__(self, state):
        self.counter = state['visited']
