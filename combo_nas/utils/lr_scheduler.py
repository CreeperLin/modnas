import torch
from .registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('lr_scheduler')


def get_lr_scheduler(optimizer, config, epochs):
    lr_type = config.type
    lr_args = config.get('args', {})
    if lr_type == 'CosineAnnealingLR':
        if 'T_max' not in lr_args:
            lr_args['T_max'] = epochs
    return build(lr_type, optimizer, **lr_args)


register(torch.optim.lr_scheduler.CosineAnnealingLR)
register(torch.optim.lr_scheduler.StepLR)
register(torch.optim.lr_scheduler.MultiStepLR)
register(torch.optim.lr_scheduler.ExponentialLR)
