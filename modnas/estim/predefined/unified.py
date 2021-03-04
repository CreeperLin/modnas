"""Unified Estimator."""
import itertools
from ..base import EstimBase
from ...core.param_space import ParamSpace
from modnas.registry.estim import register


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
        ParamSpace().update_params(params)
        n_train_batch = self.get_num_train_batch()
        n_valid_batch = self.get_num_valid_batch()
        train_epochs = self.train_epochs
        train_steps = self.train_steps
        if train_steps == 0:
            train_steps = n_train_batch
        elif train_steps == -1:
            train_steps = max(round(n_train_batch / (n_valid_batch or 1)), 1)
        if self.reset_training:
            self.reset_trainer(epochs=train_epochs)
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
        return ret

    def run_epoch(self, optim, epoch, tot_epochs):
        logger = self.logger
        config = self.config
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        train_steps = self.train_steps
        n_epoch_steps = 1 if train_steps == 0 else (self.get_num_train_batch() + train_steps - 1) // train_steps
        if self.cur_epoch >= tot_epochs:
            return 1
        # arch step
        if not optim.has_next():
            logger.info('Search: finished')
            return 1
        if self.cur_epoch >= arch_epoch_start and (self.cur_epoch - arch_epoch_start) % arch_epoch_intv == 0:
            optim.step(self)
        self.inputs = optim.next(batch_size=arch_batch_size)
        self.clear_buffer()
        self.batch_best = None
        for params in self.inputs:
            # estim step
            self.stepped(params)
        self.wait_done()
        if (epoch + 1) % n_epoch_steps != 0:
            return
        for _, res, arch_desc in self.buffer():
            score = self.get_score(res)
            if self.best_score is None or (score is not None and score > self.best_score):
                self.best_score = score
                self.best_arch_desc = arch_desc
        # save
        if config.save_arch_desc:
            self.save_arch_desc(epoch)
        if config.save_freq != 0 and self.cur_epoch % config.save_freq == 0:
            self.save_checkpoint()
        self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
        self.cur_epoch += 1

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        config = self.config
        tot_epochs = config.epochs
        self.cur_epoch += 1
        for epoch in itertools.count(0):
            if self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) == 1:
                break
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
