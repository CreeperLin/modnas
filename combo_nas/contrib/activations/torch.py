import torch.nn as nn
from combo_nas.arch_space.ops import register_op

# torch activations
register_op(nn.ELU, 'ELU')
register_op(nn.Hardshrink, 'Hardshrink')
register_op(nn.Hardtanh, 'Hardtanh')
register_op(nn.LeakyReLU, 'LeakyReLU')
register_op(nn.LogSigmoid, 'LogSigmoid')
# register_op(torch.nn.MultiheadAttention, 'MultiheadAttention')
register_op(nn.PReLU, 'PReLU')
register_op(nn.ReLU, 'ReLU')
register_op(nn.ReLU6, 'ReLU6')
register_op(nn.RReLU, 'RReLU')
register_op(nn.SELU, 'SELU')
register_op(nn.CELU, 'CELU')
register_op(nn.Sigmoid, 'Sigmoid')
register_op(nn.Softplus, 'Softplus')
register_op(nn.Softshrink, 'Softshrink')
register_op(nn.Softsign, 'Softsign')
register_op(nn.Tanh, 'Tanh')
register_op(nn.Tanhshrink, 'Tanhshrink')
register_op(nn.Threshold, 'Threshold')
