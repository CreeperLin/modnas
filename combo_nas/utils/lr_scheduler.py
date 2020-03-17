import torch
from .registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('lr_scheduler')

register(torch.optim.lr_scheduler.CosineAnnealingLR, 'cosine')
register(torch.optim.lr_scheduler.StepLR, 'step')
register(torch.optim.lr_scheduler.MultiStepLR, 'multistep')
register(torch.optim.lr_scheduler.ExponentialLR, 'exponential')
