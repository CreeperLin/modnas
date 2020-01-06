import torch
import torch.nn as nn
from combo_nas.arch_space.ops import register_op

class GELU(nn.Module):
    def __init__(self, ):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))

register_op(GELU, 'GELU')
