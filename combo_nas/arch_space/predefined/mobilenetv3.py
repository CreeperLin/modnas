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


def conv_3x3_bn(chn_in, chn_out, stride, kernel_size, use_se, use_hs):
    del use_se
    return nn.Sequential(
        nn.Conv2d(chn_in, chn_out, kernel_size, stride, kernel_size//2, bias=False),
        nn.BatchNorm2d(chn_out),
        HardSwish() if use_hs else nn.ReLU(inplace=True),
    )


def MobileInvertedConvV3(chn_in, chn_out, stride, chn, kernel_size, use_se, use_hs):
    nets = []
    if chn_in != chn:
        nets.extend([
            nn.Conv2d(chn_in, chn, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chn),
            HardSwish() if use_hs else nn.ReLU(inplace=True),
        ])
    nets.extend([
        nn.Conv2d(chn, chn, kernel_size, stride, (kernel_size - 1) // 2, groups=chn, bias=False),
        nn.BatchNorm2d(chn),
        HardSwish() if use_hs else nn.ReLU(inplace=True),
    ])
    if use_se:
        nets.append(SELayer(chn))
    nets.extend([
        nn.Conv2d(chn, chn_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(chn_out),
    ])
    return nn.Sequential(*nets)


class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, chn_in, chn, chn_out, kernel_size, stride, use_se, use_hs):
        super(MobileInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and chn_in == chn_out
        self.conv = Slot(chn_in, chn_out, stride,
                         kwargs={
                             'chn': chn,
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
    def __init__(self, chn_in, cfgs, mode, n_classes, width_mult=1., dropout_rate=0.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        block = MobileInvertedResidualBlock
        # building layers
        layers = []
        for i, (k, exp_size, c, use_se, use_hs, s) in enumerate(self.cfgs):
            chn_out = _make_divisible(c * width_mult, 8)
            if i == 0:
                # building first layer
                layers.append(conv_3x3_bn(chn_in, chn_out, s, k, use_se, use_hs))
            else:
                # building inverted residual blocks
                layers.append(block(chn_in, exp_size, chn_out, k, s, use_se, use_hs))
            chn_in = chn_out
            last_chn = exp_size
        self.features = nn.Sequential(*layers)
        # building last several layers
        last_chn = _make_divisible(last_chn * width_mult, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(chn_in, last_chn, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_chn),
            HardSwish(),
            SELayer(last_chn) if mode == 'small' else nn.Sequential()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        chn_out = 1024 if mode == 'small' else 1280
        chn_out = _make_divisible(chn_out * width_mult, 8)
        self.classifier = nn.Sequential(
            nn.Conv2d(last_chn, chn_out, kernel_size=1, stride=1),
            HardSwish(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(chn_out, n_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def get_predefined_augment_converter(self):
        return lambda slot: MobileInvertedConvV3(slot.chn_in, slot.chn_out, slot.stride,
                                                 **slot.kwargs)

    def init_model(self):
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


def mobilenetv3_large(chn_in, cfgs=None, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,   0,  16, 0, 1, 2],
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
    ] if cfgs is None else cfgs
    return MobileNetV3(chn_in, cfgs, mode='large', **kwargs)


def mobilenetv3_small(chn_in, cfgs=None, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s 
        [3,   0,  16, 0, 1, 2],
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
    ] if cfgs is None else cfgs

    return MobileNetV3(chn_in, cfgs, mode='small', **kwargs)
