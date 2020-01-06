import logging
import time
import random
import numpy as np
from ..base import ArchOptimBase
from ...utils import accuracy
from ...core.param_space import ArchParamSpace
from .gridsearch import DiscreteSpaceArchOptim

class CostModel():
    def __init__(self,):
        pass

    def fit(self, params, results):
        pass

    def predict(self, params):
        pass


class ModelOptimizer():
    def __init__(self):
        pass

    def get_maximums(self, model, size, ):
        pass


class ModelBasedArchOptim(DiscreteSpaceArchOptim):
    def __init__(self, space, cost_model, model_optimizer):
        super().__init__(space)
        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.visited = set()
        self.trials = []
        self.trial_pt = 0
        self.train_ct = 0

    def has_next(self):
        return len(self.visited) < self.space_size()

    def _next(self):
        while self.trial_pt < len(self.trials):
            index = self.trials[self.trial_pt]
            if index not in self.visited:
                break
            self.trial_pt += 1
        if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):
            index = np.random.randint(self.space_size())
            while index in self.visited:
                index = np.random.randint(self.space_size())
        return self.space.get(index)

    def next(self, batch_size):
        batch = []
        for i in range(batch_size):
            if not self.has_next():
                break
            ret = self._next()
            batch.append(ret)
            self.visited.add(ret)
        return batch

    def step(self, estim):
        for inp, res in zip(inputs, results):
            self.xs.append(inp)
            self.ys.append(res)

        if len(self.xs) >= self.plan_size * (self.train_ct + 1):
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            maximums = self.model_optimizer.find_maximums(
                self.cost_model, self.plan_size, self.visited)

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1