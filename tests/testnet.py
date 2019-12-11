import torch.nn as nn
from combo_nas.arch_space.constructor import Slot
import combo_nas.arch_space as arch_space
import combo_nas.arch_space.genotypes as gt

class TestNet(nn.Module):
    def __init__(self, chn_in, chn, n_classes, rep=10):
        super().__init__()
        self.conv1 = nn.Conv2d(chn_in, chn, 3, 1, 1)
        self.net = nn.Sequential(
            *[Slot(chn, chn, 1) for i in range(rep)]
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


@arch_space.register('TestNet')    
def get_testnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    return TestNet(chn_in, chn, n_classes)

@arch_space.register('TestAugNet')    
def get_testnet(config):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    return TestAugNet(chn_in, chn, n_classes)