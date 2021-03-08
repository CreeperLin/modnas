"""Trainer statistics reporter."""
from functools import partial
from modnas.registry.callback import register
from modnas.utils import format_value, format_dict, AverageMeter
from ..base import CallbackBase


@register
class TrainerReporter(CallbackBase):
    """Trainer statistics reporter class."""

    priority = -1

    def __init__(self, interval=0.2, format_fn=None):
        super().__init__({
            'after:TrainerBase.train_step': partial(self.report_step, 'train'),
            'after:TrainerBase.valid_step': partial(self.report_step, 'valid'),
            'after:TrainerBase.train_epoch': self.report_epoch,
            'after:TrainerBase.valid_epoch': self.report_epoch,
            'after:TrainerBase.loss': self.on_loss,
        })
        self.interval = interval
        self.fmt_fn = format_fn or {}
        self.default_fmt_fn = partial(format_value, unit=False, factor=0, prec=4)
        self.stats = None

    def init_stats(self, keys):
        """Initialize statistics."""
        self.stats = {k: AverageMeter() for k in keys}

    def reset(self):
        """Reset statistics."""
        self.stats = None

    def on_loss(self, ret, trainer, output, data, model):
        """Record batch size in each loss call."""
        self.last_batch_size = len(data[-1])

    def report_epoch(self, ret, *args, **kwargs):
        """Log statistics report in each epoch."""
        ret = ret or {}
        if self.stats:
            ret.update({k: v.avg for k, v in self.stats.items()})
        self.reset()
        return None if not ret else ret

    def report_step(self, proc, ret, trainer, estim, model, epoch, tot_epochs, step, tot_steps):
        """Log statistics report in each step."""
        if step >= tot_steps:
            return
        if step == 0:
            self.reset()
        cur_step = epoch * tot_steps + step
        interval = self.interval
        if interval and interval < 1:
            interval = int(interval * tot_steps)
        stats = ret.copy() if isinstance(ret, dict) else {}
        if self.stats is None and stats:
            self.init_stats(stats.keys())
        writer = trainer.writer
        for k, v in stats.items():
            self.stats[k].update(v, n=self.last_batch_size)
            if writer is not None:
                writer.add_scalar('/'.join(['trainer', proc, k]), v, cur_step)
        if interval is None or (interval != 0 and (step + 1) % interval == 0) or step + 1 == tot_steps:
            fmt_info = format_dict({k: self.fmt_fn.get(k, self.default_fmt_fn)(v.avg) for k, v in self.stats.items()})
            trainer.logger.info('{}: [{:3d}/{}] {}'.format(proc.title(), step + 1, tot_steps, fmt_info))
