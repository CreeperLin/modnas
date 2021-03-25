"""Estimator results exporter."""
import traceback
from modnas.registry.callback import register
from modnas.registry.export import build as build_exporter
from ..base import CallbackBase


@register
class EstimResultsExporter(CallbackBase):
    """Estimator results exporter class."""

    priority = -1

    def __init__(self, exporter='DefaultToFileExporter', file_name='results'):
        super().__init__({
            'after:EstimBase.run': self.export,
        })
        self.exporter = exporter
        self.file_name = file_name

    def export(self, ret, estim, *args, **kwargs):
        """Report ETA in each epoch."""
        path = estim.expman.join('output', self.file_name)
        estim.logger.info('Saving results to {}'.format(path))
        try:
            build_exporter(self.exporter, path=path)(ret)
        except RuntimeError:
            estim.logger.error("Failed saving results: {}".format(traceback.format_exc()))
