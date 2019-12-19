import logging
import time
import random
import numpy as np
from ..base import ArchOptimBase
from ...utils import accuracy
from ...core.param_space import ArchParamSpace

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


class ModelBasedArchOptim(ArchOptimBase):
    def __init__(self, cost_model, model_optimizer):
        super().__init__(config, None)
        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
    
    def _next(self):
        pass

    def next(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.has_next():
                batch.append(self._next())
        return batch
    
    def step(self, estim):
        pass