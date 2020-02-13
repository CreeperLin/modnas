from .darts import build_from_config as darts_builder
from .proxylessnas import build_from_config as proxylessnas_builder
from .proxylessnas import build_eas_net
from .pyramidnet import PyramidNet
from .mobilenetv2 import imagenet_mobilenetv2, cifar_mobilenetv2
from .mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from . import resnet

from .. import register
register(darts_builder, 'DARTS')
register(proxylessnas_builder, 'ProxylessNAS')
register(build_eas_net, 'PathLevelEAS')
register(PyramidNet, 'PyramidNet')
register(imagenet_mobilenetv2, 'ImageNet_MobileNetV2')
register(cifar_mobilenetv2, 'CIFAR_MobileNetV2')
register(mobilenetv3_small, 'MobileNetV3_small')
register(mobilenetv3_large, 'MobileNetV3_large')
