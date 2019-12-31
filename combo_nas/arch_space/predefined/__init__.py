from .darts import build_from_config as darts_builder
from .proxylessnas import build_from_config as proxylessnas_builder
from .proxylessnas import build_eas_net
from .pyramidnet import build_from_config as pyramidnet_builder
from .mobilenetv2 import mobilenetv2
from .mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from . import resnet

from .. import register_arch_space
register_arch_space(darts_builder, 'DARTS')
register_arch_space(proxylessnas_builder, 'ProxylessNAS')
register_arch_space(build_eas_net, 'PathLevelEAS')
register_arch_space(pyramidnet_builder, 'PyramidNet')
register_arch_space(mobilenetv2, 'MobileNetV2')
register_arch_space(mobilenetv3_small, 'MobileNetV3_small')
register_arch_space(mobilenetv3_large, 'MobileNetV3_large')
