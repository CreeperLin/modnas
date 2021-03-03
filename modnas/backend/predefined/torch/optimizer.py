"""Parameter Optimizer."""
import torch
from modnas.registry import parse_spec
from modnas.registry.optimizer import register, build


def get_optimizer(params, config, trainer_config=None):
    """Return a new Optimizer."""
    trainer_config = trainer_config or {}
    optim_type, optim_args = parse_spec(config)
    device_ids = trainer_config.get('device', [None])
    n_parallel = len(device_ids)
    if trainer_config.get('scale_lr', True) and 'lr' in optim_args:
        optim_args['lr'] *= n_parallel
    optimizer = build(optim_type, params, **optim_args)
    if n_parallel > 1:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids).module
    return optimizer


module = torch.optim

for name, attr in module.__dict__.items():
    if name.startswith('__'):
        continue
    if not callable(attr):
        continue
    if name.islower():
        continue
    if name == 'Optimizer':
        continue
    register(attr)
