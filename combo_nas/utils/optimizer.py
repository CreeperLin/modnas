import torch
from .registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('optimizer')

register(torch.optim.Adam, 'Adam')
register(torch.optim.SGD, 'SGD')
