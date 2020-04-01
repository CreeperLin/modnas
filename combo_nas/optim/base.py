""" optim base """
import random
from ..utils import get_optimizer

class OptimBase():
    def __init__(self, space, logger=None):
        self.space = space
        self.logger = logger

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass

    def has_next(self):
        pass

    def _next(self):
        pass

    def next(self, batch_size):
        batch = []
        for _ in range(batch_size):
            if not self.has_next():
                break
            batch.append(self._next())
        return batch

    def step(self, estim):
        pass

    def update(self, estim):
        return self.step(estim)


class GradientBasedOptim(OptimBase):
    def __init__(self, space, a_optim, logger=None):
        super().__init__(space, logger)
        self.a_optim = get_optimizer(self.space.tensor_values(), a_optim)

    def state_dict(self):
        return {
            'a_optim': self.a_optim.state_dict()
        }

    def load_state_dict(self, sd):
        self.a_optim.load_state_dict(sd['a_optim'])

    def optim_step(self):
        self.a_optim.step()
        self.space.on_update_tensor_params()

    def optim_reset(self):
        self.a_optim.zero_grad()


class CategoricalSpaceOptim(OptimBase):
    def __init__(self, space, logger=None):
        super().__init__(space, logger)
        self.space_size = self.space.categorical_size
        self.visited = set()

    def has_next(self):
        return len(self.visited) < self.space_size()

    def get_random_index(self):
        index = random.randint(0, self.space_size() - 1)
        while index in self.visited:
            index = random.randint(0, self.space_size() - 1)
        return index

    def is_visited(self, idx):
        return idx in self.visited

    def set_visited(self, idx):
        self.visited.add(idx)

    def get_random_params(self):
        return self.space.get_categorical_params(self.get_random_index())

    def is_visited_params(self, params):
        return self.is_visited(self.space.get_categorical_index(params))

    def set_visited_params(self, params):
        self.visited.add(self.space.get_categorical_index(params))

    def _next(self):
        raise NotImplementedError

    def next(self, batch_size):
        batch = []
        for _ in range(batch_size):
            if not self.has_next(): break
            index = self._next()
            batch.append(index)
        return batch
