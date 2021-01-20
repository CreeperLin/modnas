import logging
import torch
import numpy as np
from . import register
from ..slot import Slot
from ...core.param_space import ArchParamSpace


def parse_device(device):
    """Return device ids from config."""
    if not isinstance(device, str):
        return []
    device = device.lower()
    if device in ['cpu', 'nil', 'none']:
        return []
    if device == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in device.split(',')]


def init_device(device=None, seed=11235):
    """Initialize device and set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


@register
class DefaultInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None, device=None):
        self.seed = seed
        self.device = device

    def __call__(self, model):
        """Run constructor."""
        Slot.reset()
        ArchParamSpace.reset()
        seed = self.seed
        if seed:
            init_device(self.device, seed)
        return model


@register
class ToDevice():
    """Constructor that moves model to some device."""

    def __init__(self, device='all', data_parallel=True):
        device_ids = parse_device(device) or [None]
        self.device_ids = device_ids
        self.data_parallel = data_parallel

    def __call__(self, model):
        """Run constructor."""
        if model is None:
            return
        device_ids = self.device_ids
        if device_ids[0] is not None:
            torch.cuda.set_device(device_ids[0])
        model.to(device=device_ids[0])
        if self.data_parallel and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model


@register
class DefaultTorchCheckpointLoader():
    """Constructor that loads model checkpoints."""

    def __init__(self, path):
        logging.info('Loading torch checkpoint from {}'.format(path))
        self.chkpt = torch.load(path)

    def __call__(self, model):
        """Run constructor."""
        model.load_state_dict(self.chkpt)
        return model
