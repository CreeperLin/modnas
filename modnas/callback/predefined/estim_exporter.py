"""Estimator results exporter."""
from modnas.registry.callback import register
from ..base import CallbackBase


@register
class EstimResultsExporter(CallbackBase):
    """Estimator results exporter class."""

    priority = -1

    def __init__(self, run_file_name='results', best_file_name='best', score_fn=None,
                 chkpt_intv=0, desc_intv=0, save_chkpt_best=True, save_desc_best=True):
        super().__init__({
            'before:EstimBase.run': self.reset,
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run': self.export_run,
            'after:EstimBase.run_epoch': self.export_epoch,
        })
        self.run_file_name = run_file_name
        self.best_file_name = best_file_name
        self.score_fn = score_fn
        self.chkpt_intv = chkpt_intv
        self.desc_intv = desc_intv
        self.save_chkpt_best = save_chkpt_best
        self.save_desc_best = save_desc_best
        self.best_score = None
        self.best_arch_desc = None

    def reset(self, *args, **kwargs):
        """Reset callback states."""
        self.best_score = None
        self.best_arch_desc = None

    def on_step_done(self, ret, estim, params, value, arch_desc=None):
        """Export result on each step."""
        score = (self.score_fn or estim.get_score)(value)
        if not isinstance(score, (float, int)):
            return
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            if self.save_chkpt_best:
                estim.save_checkpoint(save_name=self.best_file_name)
            if params is not None:
                arch_desc = arch_desc or estim.get_arch_desc()
                if self.save_desc_best:
                    estim.save_arch_desc(save_name=self.best_file_name, arch_desc=arch_desc)
                self.best_arch_desc = arch_desc

    def export_run(self, ret, estim, *args, **kwargs):
        """Export results after run."""
        estim.save_arch_desc(save_name=self.run_file_name, arch_desc=ret)
        best_res = {
            'best_score': self.best_score,
        }
        if self.best_arch_desc:
            best_res['best_arch_desc'] = self.best_arch_desc
        ret = ret or {}
        ret.update(best_res)
        return ret

    def export_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Export results in each epoch."""
        if epoch >= tot_epochs:
            return
        if self.desc_intv and epoch % self.desc_intv == 0:
            estim.save_arch_desc(epoch)
        if self.chkpt_intv and epoch % self.chkpt_intv == 0:
            estim.save_checkpoint(epoch)
