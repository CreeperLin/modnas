"""Default Torch Exporters."""
import traceback
import torch
from modnas.registry.export import register
from modnas.utils.logging import get_logger


logger = get_logger('export')


@register
class DefaultTorchCheckpointExporter():
    """Exporter that saves model checkpoint to file."""

    def __init__(self, path, zip_file=None):
        self.path = path
        save_kwargs = {}
        if zip_file is not None and int('.'.join(torch.__version__.split('.'))) >= 140:
            save_kwargs['_use_new_zipfile_serialization'] = zip_file
        self.save_kwargs = save_kwargs

    def __call__(self, model):
        """Run Exporter."""
        logger.info('Saving torch checkpoint to {}'.format(self.path))
        try:
            torch.save(model.state_dict(), self.path, **self.save_kwargs)
        except RuntimeError:
            logger.error('Failed saving checkpoint: {}'.format(traceback.format_exc()))
        return model
