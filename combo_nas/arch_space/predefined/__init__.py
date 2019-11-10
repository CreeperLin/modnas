from .darts import build_from_config as darts_builder
from .proxylessnas import build_from_config as proxylessnas_builder
from .proxylessnas import build_eas_net
from .pyramidnet import build_from_config as pyramidnet_builder
from .resnet import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from .mobilenetv2 import mobilenetv2
from ...utils.registration import Registry, build, get_builder, register, register_wrapper
from functools import partial

arch_space_registry = Registry('arch_space')
register_arch_space = partial(register, arch_space_registry)
get_arch_space_builder = partial(get_builder, arch_space_registry)
build_arch_space = partial(build, arch_space_registry)
register = partial(register_wrapper, arch_space_registry)

register_arch_space(darts_builder, 'DARTS')
register_arch_space(proxylessnas_builder, 'ProxylessNAS')
register_arch_space(build_eas_net, 'PathLevelEAS')
register_arch_space(pyramidnet_builder, 'PyramidNet')
register_arch_space(resnet10, 'ResNet-10')
register_arch_space(resnet18, 'ResNet-18')
register_arch_space(resnet34, 'ResNet-34')
register_arch_space(resnet50, 'ResNet-50')
register_arch_space(resnet101, 'ResNet-101')
register_arch_space(resnet152, 'ResNet-152')
register_arch_space(resnext50_32x4d, 'ResNext50')
register_arch_space(resnext101_32x8d, 'ResNext101')
register_arch_space(mobilenetv2, 'MobileNetV2')