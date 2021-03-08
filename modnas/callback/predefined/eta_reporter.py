"""ETA (remaining time) reporter."""
from modnas.utils import ETAMeter
from modnas.registry.callback import register
from ..base import CallbackBase


@register
class ETAReporter(CallbackBase):
    """ETA reporter class."""

    priority = -1

    def __init__(self):
        super().__init__({
            'before:EstimBase.run': self.init,
            'after:EstimBase.run_epoch': self.report_epoch,
        })
        self.eta_m = None

    def init(self, estim, *args, **kwargs):
        """Initialize ETA meter."""
        tot_epochs = estim.config.get('epochs', 0)
        if tot_epochs < 1:
            return
        self.eta_m = ETAMeter(tot_epochs, estim.cur_epoch)
        self.eta_m.start()

    def report_epoch(self, ret, estim, *args, **kwargs):
        """Report ETA in each epoch."""
        if self.eta_m is None:
            return
        self.eta_m.step()
        estim.stats['ETA'] = self.eta_m.eta_fmt()
