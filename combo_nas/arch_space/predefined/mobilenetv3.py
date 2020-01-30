# modified from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import math
import torch.nn as nn
from ...arch_space.constructor import Slot

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        HardSwish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HardSwish()
    )


def MobileInvertedConvV3(inp, oup, stride, C, kernel_size, use_se, use_hs):
    if inp == C:
        return nn.Sequential(
            # dw
            nn.Conv2d(C, C, kernel_size, stride, (kernel_size - 1) // 2, groups=C, bias=False),
            nn.BatchNorm2d(C),
            HardSwish() if use_hs else nn.ReLU(inplace=True),
            # Squeeze-and-Excite
            SELayer(C) if use_se else nn.Sequential(),
            # pw-linear
            nn.Conv2d(C, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    else:
        return nn.Sequential(
            # pw
            nn.Conv2d(inp, C, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C),
            HardSwish() if use_hs else nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(C, C, kernel_size, stride, (kernel_size - 1) // 2, groups=C, bias=False),
            nn.BatchNorm2d(C),
            # Squeeze-and-Excite
            SELayer(C) if use_se else nn.Sequential(),
            HardSwish() if use_hs else nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(C, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, C, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.conv = Slot(inp, oup, stride,
                         kwargs={
                             'C': C,
                             'kernel_size': kernel_size,
                             'use_se': use_se,
                             'use_hs': use_hs
                         })

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, chn_in, cfgs, mode, num_classes, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(chn_in, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
            last_chn = exp_size * width_mult
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, _make_divisible(last_chn, 8)),
            SELayer(_make_divisible(last_chn, 8)) if mode == 'small' else nn.Sequential()
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            HardSwish()
        )
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Linear(_make_divisible(last_chn, 8), output_channel),
            nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
            HardSwish(),
            nn.Linear(output_channel, num_classes),
            nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
            HardSwish() if mode == 'small' else nn.Sequential()
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_predefined_augment_converter(self):
        return lambda slot: MobileInvertedConvV3(slot.chn_in, slot.chn_out, slot.stride,
                                                **slot.kwargs)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(chn_in, n_classes, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,  16,  16, 0, 0, 1],
        [3,  64,  24, 0, 0, 2],
        [3,  72,  24, 0, 0, 1],
        [5,  72,  40, 1, 0, 2],
        [5, 120,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [3, 240,  80, 0, 1, 2],
        [3, 200,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1]
    ]
    return MobileNetV3(chn_in, cfgs, mode='large', num_classes=n_classes, **kwargs)


def mobilenetv3_small(chn_in, n_classes, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,  16,  16, 1, 0, 2],
        [3,  72,  24, 0, 0, 2],
        [3,  88,  24, 0, 0, 1],
        [5,  96,  40, 1, 1, 2],
        [5, 240,  40, 1, 1, 1],
        [5, 240,  40, 1, 1, 1],
        [5, 120,  48, 1, 1, 1],
        [5, 144,  48, 1, 1, 1],
        [5, 288,  96, 1, 1, 2],
        [5, 576,  96, 1, 1, 1],
        [5, 576,  96, 1, 1, 1],
    ]

    return MobileNetV3(chn_in, cfgs, mode='small', num_classes=n_classes, **kwargs)
