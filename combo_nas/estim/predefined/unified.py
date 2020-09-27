"""Unified Estimator."""
import itertools
from ..base import EstimBase
from ... import utils
from ...core.param_space import ArchParamSpace
from .. import register


@register
class UnifiedEstim(EstimBase):
    """Unified Estimator class."""

    def __init__(self, train_epochs=1, train_steps=-1, reset_training=False, eval_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if train_steps != 0:
            train_epochs = 1
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.reset_training = reset_training
        self.eval_steps = eval_steps
        self.cur_step = -1
        self.best_score = None
        self.best_arch_desc = None

    def step(self, params):
        """Return evaluation results of a parameter set."""
        ArchParamSpace.update_params(params)
        n_train_batch = self.get_num_train_batch()
        n_valid_batch = self.get_num_valid_batch()
        train_epochs = self.train_epochs
        train_steps = self.train_steps
        if train_steps == 0:
            train_steps = n_train_batch
        elif train_steps == -1:
            train_steps = max(round(n_train_batch / (n_valid_batch or 1)), 1)
        if self.reset_training:
            self.reset_trainer(tot_epochs=train_epochs)
        for epoch in range(train_epochs):
            for _ in range(train_steps):
                self.cur_step += 1
                if self.cur_step >= n_train_batch:
                    self.cur_step = -1
                    break
                self.train_step(model=self.model,
                                epoch=epoch,
                                tot_epochs=train_epochs,
                                step=self.cur_step,
                                tot_steps=n_train_batch)
        if (self.cur_step + 1) % self.eval_steps != 0:
            return {'default': None}
        arch_desc = self.exporter(self.model)
        ret = self.compute_metrics()
        self.logger.info('Evaluate: {} -> {}'.format(arch_desc, ret))
        score = self.get_score(ret)
        if self.best_score is None or (score is not None and score > self.best_score):
            self.best_score = score
            self.best_arch_desc = arch_desc
        return ret

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        train_steps = self.train_steps
        self.cur_epoch += 1
        eta_m = utils.ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch_step in itertools.count(0):
            n_epoch_steps = 1 if train_steps == 0 else (self.get_num_train_batch() + train_steps - 1) // train_steps
            epoch = self.cur_epoch
            if epoch >= tot_epochs:
                break
            # arch step
            if not optim.has_next():
                logger.info('Search: finished')
                break
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            batch_best = None
            for params in self.inputs:
                # estim step
                result = self.step(params)
                self.results.append(result)
                val_score = self.get_score(result)
                if batch_best is None or val_score > batch_best:
                    batch_best = val_score
            if (epoch_step + 1) % n_epoch_steps != 0:
                continue
            # save
            if config.save_arch_desc:
                self.save_arch_desc()
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint()
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            eta_m.step()
            logger.info('Search: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                self.cur_epoch + 1, tot_epochs, batch_best or 0, self.best_score or 0, eta_m.eta_fmt()))
            self.cur_epoch += 1
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
