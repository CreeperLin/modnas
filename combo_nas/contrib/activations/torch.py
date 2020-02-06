import torch.nn as nn
from combo_nas.arch_space.ops import register

# torch activations
register(nn.ELU, 'ELU')
register(nn.Hardshrink, 'Hardshrink')
register(nn.Hardtanh, 'Hardtanh')
register(nn.LeakyReLU, 'LeakyReLU')
register(nn.LogSigmoid, 'LogSigmoid')
# register(torch.nn.MultiheadAttention, 'MultiheadAttention')
register(nn.PReLU, 'PReLU')
register(nn.ReLU, 'ReLU')
register(nn.ReLU6, 'ReLU6')
register(nn.RReLU, 'RReLU')
register(nn.SELU, 'SELU')
register(nn.CELU, 'CELU')
register(nn.Sigmoid, 'Sigmoid')
register(nn.Softplus, 'Softplus')
register(nn.Softshrink, 'Softshrink')
register(nn.Softsign, 'Softsign')
register(nn.Tanh, 'Tanh')
register(nn.Tanhshrink, 'Tanhshrink')
register(nn.Threshold, 'Threshold')
