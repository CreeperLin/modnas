# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.registration import Registry, build, get_builder, register_wrapper
from ..utils import get_same_padding
from functools import partial

op_registry = Registry('op')

def register_op(net_builder, rid=None, abbr=None):
    op_registry.register(net_builder, rid)
    if not abbr is None: op_registry.register(net_builder, abbr)
    logging.debug('registered op: {} {}'.format(rid, abbr))

def update_op(ops):
    op_registry.update(ops)

get_op_builder = partial(get_builder, op_registry)
build_op = partial(build, op_registry)
register = partial(register_wrapper, op_registry)

register_op(lambda C_in, C_out, stride: Zero(C_in, C_out, stride), 'none', 'NIL')
register_op(lambda C_in, C_out, stride: PoolBN('avg', C_in, C_out, 3, stride, 1), 'avg_pool', 'AVG')
register_op(lambda C_in, C_out, stride: PoolBN('max', C_in, C_out, 3, stride, 1), 'max_pool', 'MAX')
register_op(lambda C_in, C_out, stride: Identity() if C_in == C_out and stride == 1 
                                        else FactorizedReduce(C_in, C_out), 'skip_connect', 'IDT')
kernel_sizes = [1, 3, 5, 7, 9]
for k in kernel_sizes:
    p = get_same_padding(k)
    p2 = get_same_padding(2*k-1)
    p3 = get_same_padding(3*k-2)
    kstr = '_{k}x{k}'.format(k=k)
    kabbr = str(k)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: PoolBN('avg', C_in, C_out, ks, stride, pd), 'avg_pool'+kstr, 'AP'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: PoolBN('max', C_in, C_out, ks, stride, pd), 'max_pool'+kstr, 'MP'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: SepConv(C_in, C_out, ks, stride, pd), 'sep_conv'+kstr, 'SC'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: SepSingle(C_in, C_out, ks, stride, pd), 'sep_sing'+kstr, 'SS'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: StdConv(C_in, C_out, ks, stride, pd), 'std_conv'+kstr, 'NC'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p2: DilConv(C_in, C_out, ks, stride, pd, 2), 'dil_conv'+kstr, 'DC'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p3: DilConv(C_in, C_out, ks, stride, pd, 3), 'dil2_conv'+kstr, 'DD'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: FacConv(C_in, C_out, ks, stride, pd), 'fac_conv_{}'.format(k), 'FC'+kabbr)
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: MBConv(C_in, C_out, ks, stride, pd, 1), 'mbconv_{}_e1'.format(k), 'MB{}E1'.format(k))
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: MBConv(C_in, C_out, ks, stride, pd, 3), 'mbconv_{}_e3'.format(k), 'MB{}E3'.format(k))
    register_op(lambda C_in, C_out, stride, ks=k, pd=p: MBConv(C_in, C_out, ks, stride, pd, 6), 'mbconv_{}_e6'.format(k), 'MB{}E6'.format(k))

OPS_ORDER = ['bn','act','weight']
AFFINE = True

def configure_ops(config):
    global OPS_ORDER
    OPS_ORDER = config.ops_order.split('_')
    logging.info('configure ops: ops order set to: {}'.format(OPS_ORDER))

    global AFFINE
    AFFINE = config.affine
    logging.info('configure ops: affine: {}'.format(AFFINE))

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, expansion):
        super().__init__()
        C_t = C_in * expansion
        nets = [] if expansion == 1 else [
            nn.Conv2d(C_in, C_t, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_t, AFFINE),
            nn.ReLU6(),
        ]
        nets.extend([
            nn.Conv2d(C_t, C_t, kernel_size, stride, padding, groups=C_t, bias=False),
            nn.BatchNorm2d(C_t, AFFINE),
            nn.ReLU6(),
            nn.Conv2d(C_t, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, AFFINE)
        ])
        self.net=nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, padding):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
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
            if i=='bn':
                nets.append(nn.BatchNorm2d(C_in, affine=AFFINE))
            elif i=='weight':
                nets.append(pool)
            elif i=='act':
                pass

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, (padding, 0), bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, (1, kernel_length), 1, (0, padding), bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1)
        )

    def forward(self, x):
        return self.net(x)


class SepSingle(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1)
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=AFFINE))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
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
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
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

