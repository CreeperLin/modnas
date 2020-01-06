import torch.nn as nn
from combo_nas.arch_space.constructor import Slot
import combo_nas.arch_space as arch_space
from combo_nas.arch_space.ops import build_op
from combo_nas.arch_space.mixed_ops import build_mixed_op

class TestNet(nn.Module):
    def __init__(self, chn_in, chn, n_classes, rep=10, stage=1):
        super().__init__()
        self.conv1 = nn.Conv2d(chn_in, chn, 3, 1, 1)
        nets = []
        for i in range(stage):
            nets.extend([Slot(chn, chn, 1) for i in range(rep)])
            nets.append(nn.Conv2d(chn, 2*chn, 3, 2, 1))
            chn *= 2
        self.net = nn.Sequential(*nets)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chn, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.net(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        return x

    def get_predefined_augment_converter(self):
        return lambda slot: nn.Conv2d(slot.chn_in, slot.chn_out, kernel_size=3, stride=slot.stride,
                                      padding=1, bias=False, **slot.kwargs)

class TestAugNet(nn.Module):
    def __init__(self, chn_in, chn, n_classes, rep=10):
        super().__init__()
        self.conv1 = nn.Conv2d(chn_in, chn, 3, 1, 1)
        self.net = nn.Sequential(
            *[nn.Conv2d(chn, chn, 3, 1, 1) for i in range(rep)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chn, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.net(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        return x


class TestNetSpace(TestNet):
    def get_genotype_search_converter(self):
        ops_map = {
            'MAX': ['AVG', 'MAX'],
            'AVG': ['AVG', 'MAX'],
            'SC3': ['SC3', 'SC5', 'SC7'],
            'SC5': ['SC3', 'SC5', 'SC7'],
            'SC7': ['SC3', 'SC5', 'SC7'],
            'DC3': ['SC3', 'SC5', 'SC7'],
            'DC5': ['SC3', 'SC5', 'SC7'],
        }
        def convert_fn(slot, gene, *args, **kwargs):
            if isinstance(gene, list): gene = gene[0]
            if gene == 'NIL' or gene == 'IDT':
                return build_op(gene, slot.chn_in, slot.chn_out, slot.stride)
            else:
                return build_mixed_op(chn_in=slot.chn_in,
                                      chn_out=slot.chn_out,
                                      stride=slot.stride,
                                      ops=ops_map[gene],
                                      *args, **kwargs)
        return convert_fn


@arch_space.register('TestNet')
def get_testnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    return TestNet(chn_in, chn, n_classes)


@arch_space.register('TestNetSpace')
def get_testnet_space(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    return TestNetSpace(chn_in, chn, n_classes)


@arch_space.register('TestAugNet')
def get_testaugnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    return TestAugNet(chn_in, chn, n_classes)
