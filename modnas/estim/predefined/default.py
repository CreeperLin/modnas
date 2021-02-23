"""Estimator with default training & evaluating methods."""
import itertools
from ..base import EstimBase
from ...utils import ETAMeter
from .. import register


@register
class DefaultEstim(EstimBase):
    """Default Estimator class."""

    def __init__(self, *args, save_best=True, valid_intv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_best = save_best
        self.best_score = None
        self.valid_intv = valid_intv

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        self.print_model_info()
        config = self.config
        tot_epochs = config.epochs
        eta_m = ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch in itertools.count(self.cur_epoch + 1):
            if epoch == tot_epochs:
                break
            # train
            self.train_epoch(epoch, tot_epochs)
            # valid
            if epoch + 1 == tot_epochs or (self.valid_intv is not None and not (epoch + 1) % self.valid_intv):
                val_score = self.get_score(self.compute_metrics())
            else:
                val_score = None
            # save
            if val_score is not None and (self.best_score is None or val_score > self.best_score):
                self.best_score = val_score
                if self.save_best:
                    self.save_checkpoint(epoch, save_name='best')
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            eta_m.step()
            self.logger.info('Default: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                epoch + 1, tot_epochs, val_score or 0, self.best_score or 0, eta_m.eta_fmt()))
        ret = {'best_score': self.best_score}
        return ret
