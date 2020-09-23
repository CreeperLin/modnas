import logging
import torch
from . import register
from ..slot import Slot
from ...core.param_space import ArchParamSpace


@register
class DefaultInitConstructor():
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, model):
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
    def __init__(self, device_ids, data_parallel=True):
        device_ids = device_ids or [None]
        self.device_ids = device_ids
        self.data_parallel = data_parallel

    def __call__(self, model):
        device_ids = self.device_ids
        model.to(device=device_ids[0])
        if self.data_parallel and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model


@register
class DefaultTorchCheckpointLoader():
    def __init__(self, path):
        logging.info('Loading torch checkpoint from {}'.format(path))
        self.chkpt = torch.load(path)

    def __call__(self, model):
        model.load_state_dict(self.chkpt)
        return model
