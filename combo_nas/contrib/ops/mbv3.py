import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from combo_nas.arch_space.ops import register

def _make_divisible(v, divisor, min_value=None):
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
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        chn = _make_divisible(channel // reduction, divisor=8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, chn, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(chn, channel, 1, 1, 0),
            HardSigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


def mbconv_v3(chn_in, chn_out, stride, expansion, kernel_size, use_se=0, use_hs=0):
    nets = []
    chn = expansion * chn_in
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


for ksize in [3, 5, 7]:
    for exp in [1, 3, 6]:
        register(partial(mbconv_v3, expansion=exp, kernel_size=ksize), 'M3B{}E{}'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 0, 1), 'M3B{}E{}H'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 1, 0), 'M3B{}E{}S'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 1, 1), 'M3B{}E{}SH'.format(ksize, exp))
