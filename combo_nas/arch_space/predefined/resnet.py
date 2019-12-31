from functools import partial
import torch.nn as nn
from ...arch_space.constructor import Slot
from .. import register_arch_space

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return Slot(in_planes, out_planes, stride, groups=groups)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    chn_init = 16

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride, groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    chn_init = 16

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (1. * base_width / self.chn_init)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, chn_in, chn, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=None, norm_layer=None, expansion=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if not expansion is None:
            block.expansion = expansion
        block.chn_init = chn

        self.chn = chn
        self.groups = groups
        self.base_width = chn // groups if width_per_group is None else width_per_group
        self.conv1 = self.get_stem(chn_in, chn, norm_layer)
        
        self.layers = nn.Sequential(*[
            self._make_layer(block, (2 ** i) * chn, layers[i], stride=(1 if i==0 else 2), norm_layer=norm_layer)
        for i in range(len(layers))])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.chn, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.chn != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.chn, planes * block.expansion, stride,),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.chn, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.chn = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.chn, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_predefined_augment_converter(self):
        return lambda slot: nn.Conv2d(slot.chn_in, slot.chn_out, kernel_size=3, stride=slot.stride,
                                            padding=1, bias=False, **slot.kwargs)


class ImageNetResNet(ResNet):
    def get_stem(self, chn_in, chn, norm_layer):
        return nn.Sequential(
            nn.Conv2d(chn_in, chn, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(chn),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class CIFARResNet(ResNet):
    def get_stem(self, chn_in, chn, norm_layer):
        return nn.Sequential(
            nn.Conv2d(chn_in, chn, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(chn),
            nn.ReLU(inplace=True),
        )


def resnet10(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [1, 1, 1, 1], num_classes=n_classes, **kwargs)
    return model


def resnet18(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [2, 2, 2, 2], num_classes=n_classes, **kwargs)
    return model


def resnet32(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [5, 5, 5], num_classes=n_classes, **kwargs)
    return model


def resnet34(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def resnet50(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, Bottleneck, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def resnet56(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [9, 9, 9], num_classes=n_classes, **kwargs)
    return model


def resnet101(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, Bottleneck, [3, 4, 23, 3], num_classes=n_classes, **kwargs)
    return model


def resnet110(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, BasicBlock, [18, 18, 18], num_classes=n_classes, **kwargs)
    return model


def resnet152(resnet_cls, config, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, Bottleneck, [3, 8, 36, 3], num_classes=n_classes, **kwargs)
    return model


def resnext50_32x4d(resnet_cls, config, pretrained=False, **kwargs):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, Bottleneck, [3, 4, 6, 3], num_classes=n_classes, groups=32, width_per_group=4, **kwargs)
    return model


def resnext101_32x8d(resnet_cls, config, pretrained=False, **kwargs):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    model = resnet_cls(chn_in, chn, Bottleneck, [3, 4, 23, 3], num_classes=n_classes, groups=32, width_per_group=8, **kwargs)
    return model

def resnet(resnet_cls, config, **kwargs):
    chn_in = config.channel_in
    chn = config.channel_init
    n_classes = config.classes
    groups = config.get('groups', 1)
    width_per_group = config.get('width_per_group', None)
    layers = config.layers
    bottleneck = config.bottleneck
    block = Bottleneck if bottleneck else BasicBlock
    model = resnet_cls(chn_in, chn, block, layers, num_classes=n_classes,
                    groups=groups, width_per_group=width_per_group, **kwargs)
    return model


for resnet_cls in [CIFARResNet, ImageNetResNet]:
    name = 'CIFAR-' if resnet_cls == CIFARResNet else 'ImageNet-'
    register_arch_space(partial(resnet10, resnet_cls), name+'ResNet-10')
    register_arch_space(partial(resnet18, resnet_cls), name+'ResNet-18')
    register_arch_space(partial(resnet32, resnet_cls), name+'ResNet-32')
    register_arch_space(partial(resnet34, resnet_cls), name+'ResNet-34')
    register_arch_space(partial(resnet50, resnet_cls), name+'ResNet-50')
    register_arch_space(partial(resnet56, resnet_cls), name+'ResNet-56')
    register_arch_space(partial(resnet101, resnet_cls), name+'ResNet-101')
    register_arch_space(partial(resnet152, resnet_cls), name+'ResNet-152')
    register_arch_space(partial(resnext50_32x4d, resnet_cls), name+'ResNeXt-50')
    register_arch_space(partial(resnext101_32x8d, resnet_cls), name+'ResNeXt-101')
    register_arch_space(partial(resnet, resnet_cls), name+'ResNet')