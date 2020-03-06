import torch.nn as nn
from combo_nas.arch_space.ops import register

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


def mbconv_v3(chn_in, chn_out, stride, expansion, kernel_size, use_se, use_hs):
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
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 0, 0), 'M3B{}E{}'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 0, 1), 'M3B{}E{}H'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 1, 0), 'M3B{}E{}S'.format(ksize, exp))
        register(lambda C_in, C_out, S, k=ksize, e=exp: mbconv_v3(C_in, C_out, S, e, k, 1, 1), 'M3B{}E{}SH'.format(ksize, exp))