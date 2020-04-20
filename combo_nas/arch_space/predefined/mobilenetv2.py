import math
from collections import OrderedDict
import torch.nn as nn
from ...arch_space.constructor import Slot, default_mixed_op_converter, default_genotype_converter
from ..ops import get_same_padding
from ..ops import register as register_ops

kernel_sizes = [3, 5, 7, 9]
expand_ratios = [1, 3, 6, 9]
for k in kernel_sizes:
    for e in expand_ratios:
        p = get_same_padding(k)
        builder = lambda C_in, C_out, stride, ks=k, exp=e, pd=p: MobileInvertedConv(C_in, C_out, C_in*exp, stride, ks, pd)
        register_ops(builder, 'MB{}E{}'.format(k, e))


def round_filters(filters, width_coeff, divisor, min_depth=None):
    multiplier = width_coeff
    if not multiplier:
        return filters
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coeff):
    multiplier = depth_coeff
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MobileInvertedConv(nn.Sequential):
    def __init__(self, chn_in, chn_out, C, stride, kernel_size=3, padding=1, activation=nn.ReLU6):
        nets = [] if chn_in == C else [
            nn.Conv2d(chn_in, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            activation(inplace=True),
        ]
        nets.extend([
            nn.Conv2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=C),
            nn.BatchNorm2d(C),
            activation(inplace=True),
            nn.Conv2d(C, chn_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(chn_out)
        ])
        super().__init__(*nets)


class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, chn_in, chn_out, stride=1, t=6, activation=nn.ReLU6):
        super(MobileInvertedResidualBlock, self).__init__()
        self.stride = stride
        self.t = t
        self.chn_in = chn_in
        self.chn_out = chn_out
        C = chn_in * t
        self.conv = Slot(chn_in, chn_out, stride, kwargs={'C': C, 'activation': activation})

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.stride == 1 and self.chn_in == self.chn_out:
            out += residual
        return out

class MobileNetV2(nn.Module):

    def __init__(self, chn_in, n_classes, cfgs, width_coeff=1.0, depth_coeff=1.0,
                 resolution=None, dropout_rate=0.2, activation=nn.ReLU6):
        del resolution
        super(MobileNetV2, self).__init__()
        self.activation = activation
        self.n_classes = n_classes

        divisor = 8
        self.t = [cfg[0] for cfg in cfgs]
        self.c = [round_filters(cfg[1], width_coeff, divisor) for cfg in cfgs]
        self.n = [round_repeats(cfg[2], depth_coeff) for cfg in cfgs]
        self.s = [cfg[3] for cfg in cfgs]
        self.conv_first = nn.Sequential(
            nn.Conv2d(chn_in, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1),
            nn.BatchNorm2d(self.c[0]),
            self.activation(inplace=True),
        )
        self.bottlenecks = self._make_bottlenecks()

        self.last_conv_out_ch = round_filters(1280, width_coeff, divisor)
        self.conv_last = nn.Sequential(
            nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_conv_out_ch),
            self.activation(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(self.last_conv_out_ch, self.n_classes)

    def _make_stage(self, chn_in, chn_out, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "MobileInvertedResidualBlock_{}".format(stage)
        for i in range(n):
            # First module is the only one utilizing stride
            s = stride if i==0 else 1
            name = stage_name + "_{}".format(i)
            module = MobileInvertedResidualBlock(chn_in=chn_in, chn_out=chn_out, stride=s, t=t,
                                      activation=self.activation)
            modules[name] = module
            chn_in = chn_out
        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"
        for i in range(0, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(chn_in=self.c[i], chn_out=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t[i + 1], stage=i)
            modules[name] = module
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_predefined_augment_converter(self):
        return lambda slot: MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)

    def get_predefined_search_converter(self, fix_first=True, add_zero_op=True):
        def convert_fn(slot, primitives, *args, **kwargs):
            if convert_fn.fix_first and not hasattr(convert_fn, 'first'):
                ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
                convert_fn.first = True
            else:
                primitives = primitives[:]
                if convert_fn.add_zero_op and slot.stride == 1 and slot.chn_in == slot.chn_out:
                    primitives.append('NIL')
                ent = default_mixed_op_converter(slot, primitives=primitives, *args, **kwargs)
            return ent
        convert_fn.fix_first = fix_first
        convert_fn.add_zero_op = add_zero_op
        return convert_fn

    def get_genotype_augment_converter(self, fix_first=True):
        def convert_fn(slot, *args, **kwargs):
            if convert_fn.fix_first and not hasattr(convert_fn, 'first'):
                ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
                convert_fn.first = True
            else:
                ent = default_genotype_converter(slot, *args, **kwargs)
            return ent
        convert_fn.fix_first = fix_first
        return convert_fn


def imagenet_mobilenetv2(chn_in, n_classes, cfgs=None, **kwargs):
    default_cfgs = [
        # t, c, n, s,
        [0, 32, 1, 2],
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
    if cfgs is None:
        cfgs = default_cfgs
    return MobileNetV2(chn_in, n_classes, cfgs, **kwargs)


def cifar_mobilenetv2(chn_in, n_classes, cfgs=None, **kwargs):
    default_cfgs = [
        # t, c, n, s,
        [0, 32, 1, 1], # stride = 1
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 1], # stride = 1
        [6, 320, 1, 1]
    ]
    if cfgs is None:
        cfgs = default_cfgs
    return MobileNetV2(chn_in, n_classes, cfgs, **kwargs)


def mobilenetv2(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2(chn_in, n_classes, cfgs, **kwargs)
