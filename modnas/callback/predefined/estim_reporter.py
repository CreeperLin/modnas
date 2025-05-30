"""Estimator statistics reporter."""
from modnas.registry.callback import register
from modnas.utils import format_dict
from ..base import CallbackBase


@register
class EstimReporter(CallbackBase):
    """Estimator statistics reporter class."""

    priority = -10

    def __init__(self, interval=None, format_fn=None):
        super().__init__({
            'after:EstimBase.run_epoch': self.report_epoch,
        })
        self.interval = interval
        self.format_fn = format_fn

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Log statistics report in each epoch."""
        if epoch >= tot_epochs:
            return
        interval = self.interval
        if interval and interval < 1:
            interval = int(interval * tot_epochs)
        stats = ret.copy() if isinstance(ret, dict) else {}
        if interval is None or (interval != 0 and (epoch + 1) % interval == 0) or epoch + 1 == tot_epochs:
            fmt_info = format_dict(stats, fmt_val=self.format_fn)
            estim.logger.info('[{:3d}/{}] {}'.format(epoch + 1, tot_epochs, fmt_info))
        return ret
