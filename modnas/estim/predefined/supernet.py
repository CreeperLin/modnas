"""Supernet-based Estimator."""
import itertools
from ..base import EstimBase
from ...core.param_space import ArchParamSpace
from ...utils import ETAMeter
from .. import register


@register
class SuperNetEstim(EstimBase):
    """Supernet-based Estimator class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_score = None
        self.best_arch_desc = None

    def print_tensor_params(self, max_num=3):
        """Log current tensor parameter values."""
        logger = self.logger
        ap_cont = tuple(a.detach().softmax(dim=-1).cpu().numpy() for a in ArchParamSpace.tensor_values())
        max_num = min(len(ap_cont) // 2, max_num)
        logger.info('TENSOR: {}\n{}'.format(
            len(ap_cont), '\n'.join([str(a) for a in (ap_cont[:max_num] + ('...', ) + ap_cont[-max_num:])])))

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        model = self.model
        config = self.config
        tot_epochs = config.epochs
        eta_m = ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch in itertools.count(self.cur_epoch + 1):
            if epoch == tot_epochs:
                break
            # train
            self.print_tensor_params()
            self.search_epoch(epoch, optim)
            # eval
            arch_desc = self.exporter(model)
            mt_ret = self.compute_metrics()
            self.logger.info('Evaluate: {} -> {}'.format(arch_desc, mt_ret))
            score = self.get_score(mt_ret)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_arch_desc = arch_desc
            # save
            if config.save_arch_desc:
                self.save_arch_desc(epoch, arch_desc=arch_desc)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            eta_m.step()
            self.logger.info('Search: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                epoch + 1, tot_epochs, score or 0, self.best_score or 0, eta_m.eta_fmt()))
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }

    def search_epoch(self, epoch, optim):
        """Run search for one epoch."""
        config = self.config
        n_trn_batch = self.get_num_train_batch(epoch)
        n_val_batch = self.get_num_valid_batch(epoch)
        tot_epochs = config.epochs
        update_arch = False
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
            update_arch = True
            arch_update_intv = config.arch_update_intv
            if arch_update_intv == -1:  # update proportionally
                arch_update_intv = max(n_trn_batch / n_val_batch, 1) if n_val_batch else 1
            elif arch_update_intv == 0:  # update last step
                arch_update_intv = n_trn_batch
            arch_update_batch = config.arch_update_batch
        arch_step = 0
        for step in range(n_trn_batch):
            # optim step
            if update_arch and (step + 1) // arch_update_intv > arch_step:
                for _ in range(arch_update_batch):
                    optim.step(self)
                arch_step += 1
            # supernet step
            optim.next()
            self.trainer.train_step(estim=self,
                                    model=self.model,
                                    epoch=epoch,
                                    tot_epochs=tot_epochs,
                                    step=step,
                                    tot_steps=n_trn_batch)
