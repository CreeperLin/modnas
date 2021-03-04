from functools import partial
from modnas.registry.callback import register
from modnas.utils import format_value, format_dict
from ..base import CallbackBase


@register
class EstimReporter(CallbackBase):

    priority = -10

    def __init__(self, interval=None, format_fn=None):
        super().__init__({
            'after:EstimBase.run_epoch': self.report_epoch,
        })
        self.interval = interval
        self.fmt_fn = format_fn or {}
        self.default_fmt_fn = partial(format_value, unit=False, factor=0, prec=4, to_str=True)

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        if epoch >= tot_epochs:
            return
        interval = self.interval
        if interval and interval < 1:
            interval = int(interval * tot_epochs)
        stats = ret.copy() if isinstance(ret, dict) else {}
        stats.update(estim.stats)
        if interval is None or (interval != 0 and (epoch + 1) % interval == 0) or epoch + 1 == tot_epochs:
            fmt_info = format_dict({k: self.fmt_fn.get(k, self.default_fmt_fn)(v) for k, v in stats.items()})
            estim.logger.info('[{:3d}/{}] {}'.format(epoch + 1, tot_epochs, fmt_info))
        estim.stats = {}
