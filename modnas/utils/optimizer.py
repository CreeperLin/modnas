"""Parameter Optimizer."""
import torch
from ..registry.optimizer import register, get_builder, build, register_as


def get_optimizer(params, config, trainer_config=None):
    """Return a new Optimizer."""
    trainer_config = trainer_config or {}
    optim_type = config['type']
    optim_args = config.get('args', {})
    device_ids = trainer_config.get('device', [None])
    n_parallel = len(device_ids)
    if trainer_config.get('scale_lr', True) and 'lr' in optim_args:
        optim_args['lr'] *= n_parallel
    optimizer = build(optim_type, params, **optim_args)
    if n_parallel > 1:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids).module
    return optimizer


register(torch.optim.Adam)
register(torch.optim.SGD)
register(torch.optim.RMSprop)
