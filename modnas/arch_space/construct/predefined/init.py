"""Initialization constructor."""
import numpy as np
from modnas.registry.construct import register
from modnas.core.param_space import ParamSpace


@register
class DefaultInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, model):
        """Run constructor."""
        ParamSpace().reset()
        seed = self.seed
        if seed:
            np.random.seed(seed)
        return model
