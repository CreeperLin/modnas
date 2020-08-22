import traceback
import logging
import torch
from . import register

@register
class DefaultTorchCheckpointExporter():
    def __init__(self, path):
        self.path = path

    def __call__(self, model):
        logging.info('Saving torch checkpoint to {}'.format(self.path))
        try:
            torch.save(model.state_dict(), self.path)
        except RuntimeError:
            logging.error('Failed saving checkpoint: {}'.format(traceback.format_exc()))
        return model
