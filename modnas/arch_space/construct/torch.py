import logging
import torch
import numpy as np
from . import register
from ..slot import Slot
from ...core.param_space import ArchParamSpace


@register
class DefaultInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, model):
        """Run constructor."""
        Slot.reset()
        ArchParamSpace.reset()
        seed = self.seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return model


@register
class ToDevice():
    """Constructor that moves model to some device."""

    def __init__(self, device_ids, data_parallel=True):
        device_ids = device_ids or [None]
        self.device_ids = device_ids
        self.data_parallel = data_parallel

    def __call__(self, model):
        """Run constructor."""
        if model is None:
            return
        device_ids = self.device_ids
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
