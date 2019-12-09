import torch
import torch.nn as nn
import math
from ...arch_space.constructor import Slot

class GroupConv(nn.Module):
    def __init__(self, chn_in, chn_out, kernel_size, stride=1, padding=0, groups=1, relu=True, affine=True):
        super().__init__()
        chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        if chn_out is None:
            chn_out = chn_in
        self.chn_out = chn_out
        C = chn_in
        nets = []
        OPS_ORDER = ['bn', 'act', 'weight']
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='act' and relu:
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(chn_in, chn_out, kernel_size, stride, padding, groups=groups, bias=bias))
                C = chn_out
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        return self.net(x)

class BottleneckBlock(nn.Module):
    def __init__(self, C_in, C, stride=1, groups=1, bneck_ratio=4,
                 downsample=None):
        super(BottleneckBlock, self).__init__()
        self.bottle_in = GroupConv(C_in, C, 1, 1, 0, relu=False)
        self.cell = Slot(C, C, stride, groups=groups)
        self.bottle_out = GroupConv(C, C * bneck_ratio, 1, 1, 0)
        self.bn = nn.BatchNorm2d(C * bneck_ratio)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bottle_in(x)
        out = self.cell([out])
        out = self.bottle_out(out)
        out = self.bn(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel,
                                 featuremap_size[0], featuremap_size[1]).to(device=x.device)
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 
        return out


class PyramidNet(nn.Module):
        
    def __init__(self, chn_in, chn, n_classes, n_groups, n_blocks, conv_groups, bneck_ratio, alpha):
        super(PyramidNet, self).__init__()
        self.chn_in = chn_in
        self.chn = chn
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.n_blocks = n_blocks
        self.conv_groups = conv_groups
        self.bneck_ratio = bneck_ratio
        self.addrate = alpha / (self.n_groups*self.n_blocks*1.0)
        
        block = BottleneckBlock
        self.chn_cur = self.chn
        self.conv1 = nn.Conv2d(self.chn_in, self.chn_cur, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.chn_cur)
        self.chn_in = self.chn_cur
        
        groups = []
        for i in range(0, self.n_groups):
            stride = 1 if i==0 else 2
            groups.append(self.pyramidal_make_layer(block, self.n_blocks, stride))
        self.pyramid = nn.Sequential(*groups)
        
        self.chn_fin = int(round(self.chn_cur)) * self.bneck_ratio
        self.bn_final= nn.BatchNorm2d(self.chn_fin)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.chn_fin, self.n_classes)

    def pyramidal_make_layer(self, block, n_blocks, stride):
        downsample = None
        if stride != 1: # or self.chn_cur != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        for i in range(0, n_blocks):
            b_stride = stride if i==0 else 1
            chn_prev = int(round(self.chn_in))
            chn_next = int(round(self.chn_cur + self.addrate))
            chn_next -= chn_next % self.conv_groups
            blk = block(chn_prev, chn_next,
                        stride=b_stride,
                        groups=self.conv_groups,
                        bneck_ratio=self.bneck_ratio,
                        downsample=downsample)
            layers.append(blk)
            self.chn_cur += self.addrate
            self.chn_in = chn_next * self.bneck_ratio
            downsample = None
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pyramid(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x

    def get_default_converter(self):
        return lambda slot: GroupConv(slot.chn_in, slot.chn_out, 3, slot.stride, 1, **slot.kwargs)


def build_from_config(config):
    chn_in = config.channel_in
    chn = config.channel_init
    chn_cur = chn * config.channel_multiplier
    n_classes = config.classes
    n_groups = config.groups
    conv_groups = config.conv_groups
    n_blocks = config.blocks
    alpha = config.alpha
    pyramidnet_kwargs = {
        'chn_in': chn_in,
        'chn': chn,
        'n_classes': n_classes,
        'n_groups': n_groups,
        'n_blocks': n_blocks,
        'conv_groups': conv_groups,
        'alpha': alpha,
        'bneck_ratio': config.bottleneck_ratio,
    }
    return PyramidNet(**pyramidnet_kwargs)