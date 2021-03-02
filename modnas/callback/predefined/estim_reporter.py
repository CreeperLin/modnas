from modnas.registry.callback import register
from modnas.utils import format_value
from ..base import CallbackBase


def format_key(key):
    key = ' '.join(key.split('_'))
    return key.title() if key.islower() else key


def format_val(val):
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return format_value(val, unit=False, factor=0, prec=4)


@register
class EstimReporter(CallbackBase):

    priority = -1

    def __init__(self, interval=None):
        super().__init__({
            'after:EstimBase.run_epoch': self.report_epoch,
        })
        self.interval = interval

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        if epoch >= tot_epochs:
            return
        interval = self.interval
        if interval and interval < 1:
            interval = int(interval * tot_epochs)
        stats = ret.copy() if isinstance(ret, dict) else {}
        stats['best'] = estim.best_score
        stats.update(estim.stats)
        if interval is None or (interval != 0 and epoch % interval == 0):
            fmt_info = ' | '.join(['{}: {{{}}}'.format(format_key(k), k) for k in stats])
            fmt_info = fmt_info.format(**{k: format_val(v) for k, v in stats.items()})
            estim.logger.info('[{:3d}/{}] {}'.format(epoch + 1, tot_epochs, fmt_info))
        estim.stats = {}
