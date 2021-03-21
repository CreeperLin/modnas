"""Swish activation functions."""
import torch.nn as nn
import torch.nn.functional as F
from modnas.registry.arch_space import register


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x):
        """Return module output."""
        return x * F.sigmoid(x)


register(HardSigmoid)
register(HardSwish)
register(Swish)
