import logging
import torch
import torch.nn as nn
from ..utils import get_same_padding
from .slot import register_slot_ccs

register_slot_ccs(lambda C_in, C_out, stride: PoolBN('avg', C_in, C_out, 3, stride, 1), 'AVG')
register_slot_ccs(lambda C_in, C_out, stride: PoolBN('max', C_in, C_out, 3, stride, 1), 'MAX')
register_slot_ccs(lambda C_in, C_out, stride: Identity()
                  if C_in == C_out and stride == 1 else FactorizedReduce(C_in, C_out), 'IDT')

kernel_sizes = [1, 3, 5, 7, 9, 11, 13]
for k in kernel_sizes:
    p = get_same_padding(k)
    p2 = get_same_padding(2 * k - 1)
    p3 = get_same_padding(3 * k - 2)
    kabbr = str(k)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: PoolBN('avg', C_in, C_out, ks, stride, pd), 'AP' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: PoolBN('max', C_in, C_out, ks, stride, pd), 'MP' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: SepConv(C_in, C_out, ks, stride, pd), 'SC' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: SepSingle(C_in, C_out, ks, stride, pd), 'SS' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: StdConv(C_in, C_out, ks, stride, pd), 'NC' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p2: DilConv(C_in, C_out, ks, stride, pd, 2), 'DC' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p3: DilConv(C_in, C_out, ks, stride, pd, 3), 'DD' + kabbr)
    register_slot_ccs(lambda C_in, C_out, stride, ks=k, pd=p: FacConv(C_in, C_out, ks, stride, pd), 'FC' + kabbr)

OPS_ORDER = ['bn', 'act', 'weight']
AFFINE = True
BIAS = False
INPLACE = False


def configure_ops(ops_order='act_weight_bn', affine=False, bias=False, inplace=False):
    global OPS_ORDER
    OPS_ORDER = ops_order.split('_')
    global AFFINE
    AFFINE = affine
    global BIAS
    BIAS = False if OPS_ORDER[-1] == 'bn' else bias
    global INPLACE
    INPLACE = False if OPS_ORDER[0] == 'act' else inplace
    logging.info('ops config: ops_order: {} affine: {} bias: {} inplace: {}'.format(OPS_ORDER, AFFINE, BIAS, INPLACE))


class DropPath_(nn.Module):
    def __init__(self, prob=0.):
        super().__init__()
        self.prob = prob

    def extra_repr(self):
        return 'prob={}, inplace'.format(self.prob)

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            # per data point mask; assuming x in cuda.
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.div_(keep_prob).mul_(mask)
        return x


class PoolBN(nn.Module):
    """AvgPool or MaxPool - BN."""

    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        if C_in != C_out:
            raise ValueError('invalid channel in pooling layer')
        if pool_type.lower() == 'max':
            pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError('invalid pooling layer type')

        nets = []
        for i in OPS_ORDER:
            if i == 'bn':
                nets.append(nn.BatchNorm2d(C_in, affine=AFFINE))
            elif i == 'weight':
                nets.append(pool)
            elif i == 'act':
                pass

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class StdConv(nn.Module):
    """Standard conv, ReLU - Conv - BN."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i == 'bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i == 'weight':
                nets.append(nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=BIAS, groups=groups))
                C = C_out
            elif i == 'act':
                nets.append(nn.ReLU(inplace=INPLACE))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """Factorized conv, ReLU - Conv(Kx1) - Conv(1xK) - BN."""

    def __init__(self, C_in, C_out, kernel_length, stride, padding):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i == 'bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i == 'weight':
                nets.append(nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, (padding, 0), bias=BIAS))
                nets.append(nn.Conv2d(C_in, C_out, (1, kernel_length), 1, (0, padding), bias=BIAS))
                C = C_out
            elif i == 'act':
                nets.append(nn.ReLU(inplace=INPLACE))

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """(Dilated) depthwise separable conv.

    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i == 'bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i == 'weight':
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=BIAS))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=BIAS))
                C = C_out
            elif i == 'act':
                nets.append(nn.ReLU(inplace=INPLACE))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """Depthwise separable conv, DilConv(dilation=1) * 2."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1),
                                 DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1))

    def forward(self, x):
        return self.net(x)


class SepSingle(nn.Module):
    """Depthwise separable conv, DilConv(dilation=1) * 1."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i == 'bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i == 'weight':
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=BIAS))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=BIAS))
                C = C_out
            elif i == 'act':
                nets.append(nn.ReLU(inplace=INPLACE))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride, *args, **kwargs):
        super().__init__()
        if C_in != C_out:
            raise ValueError('invalid channel in zero layer')
        self.stride = stride
        self.C_out = C_out

    def forward(self, x):
        if self.stride == 1:
            return x * 0.
        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise(stride=2)."""

    def __init__(self, C_in, C_out):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=AFFINE)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


register_slot_ccs(Zero, 'NIL')
