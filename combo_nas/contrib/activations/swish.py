import torch.nn as nn
import torch.nn.functional as F
from combo_nas.arch_space.ops import register


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


register(HardSigmoid, 'hSigmoid')
register(HardSwish, 'hSwish')
register(Swish, 'Swish')
