import torch.nn as nn
from combo_nas.arch_space.ops import register

# torch activations
register(nn.ELU)
register(nn.Hardshrink)
register(nn.Hardtanh)
register(nn.LeakyReLU)
register(nn.LogSigmoid)
# register(torch.nn.MultiheadAttention)
register(nn.PReLU)
register(nn.ReLU)
register(nn.ReLU6)
register(nn.RReLU)
register(nn.SELU)
register(nn.CELU)
register(nn.Sigmoid)
register(nn.Softplus)
register(nn.Softshrink)
register(nn.Softsign)
register(nn.Tanh)
register(nn.Tanhshrink)
register(nn.Threshold)
