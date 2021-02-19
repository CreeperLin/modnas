"""Basic Optimizer classes."""
import random
from ..utils.optimizer import get_optimizer
from ..core.param_space import ParamSpace
from ..core.event import event_hooked_subclass


@event_hooked_subclass
class OptimBase():
    """Base Optimizer class."""

    def __init__(self, space=None, logger=None):
        self.space = space or ParamSpace()
        self.logger = logger

    def state_dict(self):
        """Return current states."""
        return {}

    def load_state_dict(self, sd):
        """Resume states."""
        pass

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        pass

    def _next(self):
        """Return the next set of parameters."""
        pass

    def next(self, batch_size=1):
        """Return the next batch of parameter sets."""
        batch = []
        for _ in range(batch_size):
            if not self.has_next():
                break
            batch.append(self._next())
        return batch

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        pass

    def update(self, estim):
        """Equal to Optimizer step."""
        return self.step(estim)


class GradientBasedOptim(OptimBase):
    """Gradient-based Optimizer class."""

    _default_optimizer_conf = {
        'type': 'Adam',
        'args': {
            'lr': 0.0003,
            'betas': [0.5, 0.999],
            'weight_decay': 0.001,
        }
    }

    def __init__(self, space=None, a_optim=None, logger=None):
        super().__init__(space, logger)
        self.a_optim = get_optimizer(self.space.tensor_values(), a_optim or GradientBasedOptim._default_optimizer_conf)

    def state_dict(self):
        """Return current states."""
        return {'a_optim': self.a_optim.state_dict()}

    def load_state_dict(self, sd):
        """Resume states."""
        self.a_optim.load_state_dict(sd['a_optim'])

    def optim_step(self):
        """Do tensor parameter optimizer step."""
        self.a_optim.step()
        self.space.on_update_tensor_params()

    def optim_reset(self):
        """Prepare tensor parameter optimizer step."""
        self.a_optim.zero_grad()

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return True

    def _next(self):
        return {}


class CategoricalSpaceOptim(OptimBase):
    """Categorical space Optimizer class."""

    def __init__(self, space=None, logger=None):
        super().__init__(space, logger)
        self.space_size = self.space.categorical_size
        self.visited = set()

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return len(self.visited) < self.space_size()

    def get_random_index(self):
        """Return a random index from categorical space."""
        index = random.randint(0, self.space_size() - 1)
        while index in self.visited:
            index = random.randint(0, self.space_size() - 1)
        return index

    def is_visited(self, idx):
        """Return True if a space index is already visited."""
        return idx in self.visited

    def set_visited(self, idx):
        """Set a space index as visited."""
        self.visited.add(idx)

    def get_random_params(self):
        """Return a random parameter set from categorical space."""
        return self.space.get_categorical_params(self.get_random_index())

    def is_visited_params(self, params):
        """Return True if a parameter set is already visited."""
        return self.is_visited(self.space.get_categorical_index(params))

    def set_visited_params(self, params):
        """Set a parameter set as visited."""
        self.visited.add(self.space.get_categorical_index(params))

    def _next(self):
        raise NotImplementedError
