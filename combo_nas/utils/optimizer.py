import torch
from .registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('optimizer')


def get_optimizer(params, config, device_ids=None, scale_lr=True):
    optim_type = config.type
    optim_args = config.get('args', {})
    n_parallel = 1 if device_ids is None else len(device_ids)
    if scale_lr and 'lr' in optim_args:
        optim_args['lr'] *= n_parallel
    optimizer = build(optim_type, params, **optim_args)
    if n_parallel > 1:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids).module
    return optimizer


register(torch.optim.Adam)
register(torch.optim.SGD)
register(torch.optim.RMSprop)
