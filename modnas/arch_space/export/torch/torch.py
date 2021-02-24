"""Default Torch Exporters."""
import traceback
import logging
import torch
from modnas.registry.export import register


@register
class DefaultTorchCheckpointExporter():
    """Exporter that saves model checkpoint to file."""

    def __init__(self, path):
        self.path = path

    def __call__(self, model):
        """Run Exporter."""
        logging.info('Saving torch checkpoint to {}'.format(self.path))
        try:
            torch.save(model.state_dict(), self.path)
        except RuntimeError:
            logging.error('Failed saving checkpoint: {}'.format(traceback.format_exc()))
        return model
