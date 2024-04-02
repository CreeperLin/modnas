"""LR Scheduler."""
import torch
from modnas.registry import parse_spec
from modnas.registry.lr_scheduler import register, build


def get_lr_scheduler(optimizer, config, trainer_config=None):
    """Return a new LR Scheduler."""
    trainer_config = trainer_config or {}
    lr_type, lr_args = parse_spec(config)
    if lr_type == 'CosineAnnealingLR':
        if 'T_max' not in lr_args and 'epochs' in trainer_config:
            lr_args['T_max'] = trainer_config['epochs']
    return build(lr_type, optimizer, **lr_args)


module = torch.optim.lr_scheduler

for name, attr in module.__dict__.items():
    if name.startswith('_'):
        continue
    if not callable(attr):
        continue
    if name.islower():
        continue
    if name in ['Optimizer', 'LRScheduler']:
        continue
    register(attr)
