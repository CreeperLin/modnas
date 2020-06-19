from ..base import MetricsBase
from .. import register_as

@register_as('Validate')
class ValidateMetrics(MetricsBase):
    def __init__(self, logger, field='acc_top1'):
        super().__init__(logger)
        self.field = field

    def compute(self, model):
        estim = self.get_estim()
        val_res = estim.validate_epoch(model=model)
        if isinstance(val_res, dict):
            if self.field is None:
                val_res = list(val_res.values())[0]
            else:
                val_res = val_res.get(self.field, None)
        return val_res
