"""Default Torch Exporters."""
import traceback
import torch
from modnas.registry.export import register
from modnas.utils.logging import get_logger


logger = get_logger('export')


@register
class DefaultTorchCheckpointExporter():
    """Exporter that saves model checkpoint to file."""

    def __init__(self, path):
        self.path = path

    def __call__(self, model):
        """Run Exporter."""
        logger.info('Saving torch checkpoint to {}'.format(self.path))
        try:
            torch.save(model.state_dict(), self.path)
        except RuntimeError:
            logger.error('Failed saving checkpoint: {}'.format(traceback.format_exc()))
        return model
