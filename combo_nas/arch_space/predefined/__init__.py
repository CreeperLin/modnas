from .proxylessnas import build_from_config as proxylessnas_builder
from .proxylessnas import build_eas_net
from .pyramidnet import PyramidNet
from .mobilenetv2 import imagenet_mobilenetv2, cifar_mobilenetv2, mobilenetv2
from .mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from . import darts
from . import resnet
from . import shufflenetv2

from .. import register
register(proxylessnas_builder, 'ProxylessNAS')
register(build_eas_net, 'PathLevelEAS')
register(PyramidNet, 'PyramidNet')
register(imagenet_mobilenetv2, 'ImageNet_MobileNetV2')
register(cifar_mobilenetv2, 'CIFAR_MobileNetV2')
register(mobilenetv2, 'MobileNetV2')
register(mobilenetv3_small, 'MobileNetV3_small')
register(mobilenetv3_large, 'MobileNetV3_large')
