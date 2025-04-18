"""Estimator-based metrics."""
from ..base import MetricsBase
from modnas.registry.metrics import register


@register
class ValidateMetrics(MetricsBase):
    """Estimator validation metrics class."""

    def __init__(self, field=None):
        super().__init__()
        self.field = field

    def __call__(self, model):
        """Return metrics output."""
        estim = self.estim
        val_res = estim.valid_epoch(model=model)
        if isinstance(val_res, dict):
            field = self.field
            default_res = list(val_res.values())[0]
            if field is None:
                val_res = default_res
            elif field in val_res:
                val_res = val_res[field]
            else:
                self.logger.error('field \"{}\" not exists, using default'.format(field))
                val_res = default_res
        return val_res
