import torch.nn as nn
from combo_nas.arch_space.ops import register_op

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        del inplace
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


register_op(HardSigmoid, 'hSigmoid')
register_op(Swish, 'Swish')
register_op(HardSwish, 'hSwish')
