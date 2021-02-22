"""LR Scheduler."""
import torch
from ..registry.lr_scheduler import register, get_builder, build, register_as


def get_lr_scheduler(optimizer, config, trainer_config=None):
    """Return a new LR Scheduler."""
    trainer_config = trainer_config or {}
    lr_type = config['type']
    lr_args = config.get('args', {})
    if lr_type == 'CosineAnnealingLR':
        if 'T_max' not in lr_args and 'epochs' in trainer_config:
            lr_args['T_max'] = trainer_config['epochs']
    return build(lr_type, optimizer, **lr_args)


register(torch.optim.lr_scheduler.CosineAnnealingLR)
register(torch.optim.lr_scheduler.StepLR)
register(torch.optim.lr_scheduler.MultiStepLR)
register(torch.optim.lr_scheduler.ExponentialLR)
