import math
import torch.nn as nn
from ...arch_space.constructor import Slot
from collections import OrderedDict

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

def MobileInvertedConv(chn_in, chn_out, C, stride, activation):
    return nn.Sequential(
        nn.Conv2d(chn_in, C, kernel_size=1, bias=False),
        nn.BatchNorm2d(C),
        activation(inplace=True),
        nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, bias=False, groups=C),
        nn.BatchNorm2d(C),
        activation(inplace=True),
        nn.Conv2d(C, chn_out, kernel_size=1, bias=False),
        nn.BatchNorm2d(chn_out)
    )

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

    def __init__(self, chn_in, n_classes, t=6,
                 width_coeff=1.0, depth_coeff=1.0, resolution=None, dropout_rate=0.2, activation=nn.ReLU6):
        del resolution
        super(MobileNetV2, self).__init__()
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.n_classes = n_classes

        base_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        base_repeats = [1, 1, 2, 3, 4, 3, 3, 1]

        divisor = 8
        self.c = [round_filters(ch, width_coeff, divisor) for ch in base_channels]
        self.n = [round_repeats(r, depth_coeff) for r in base_repeats]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(chn_in, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        self.last_conv_out_ch = round_filters(1280, width_coeff, divisor)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(self.last_conv_out_ch, self.n_classes)

    def _make_stage(self, chn_in, chn_out, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "MobileInvertedResidualBlock_{}".format(stage)

        # First module is the only one utilizing stride
        first_module = MobileInvertedResidualBlock(chn_in=chn_in, chn_out=chn_out, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more MobileInvertedResidualBlock depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = MobileInvertedResidualBlock(chn_in=chn_out, chn_out=chn_out, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(chn_in=self.c[0], chn_out=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more MobileInvertedResidualBlock depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(chn_in=self.c[i], chn_out=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_predefined_augment_converter(self):
        return lambda slot: MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
