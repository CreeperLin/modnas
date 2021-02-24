import itertools
from ..base import EstimBase
from ...core.param_space import ParamSpace
from modnas.registry.estim import register


@register
class RegressionEstim(EstimBase):
    def __init__(self, *args, predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.best_score = None
        self.best_arch_desc = None
        self.best_score_batch = None
        self.best_desc_batch = None

    def step(self, params):
        ParamSpace().update_params(params)
        arch_desc = self.get_arch_desc()
        score = self.predictor.predict(arch_desc)
        return score

    def run(self, optim):
        config = self.config
        tot_epochs = config.epochs
        logger = self.logger
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        for epoch in itertools.count(self.cur_epoch + 1):
            if epoch == tot_epochs:
                break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            inputs = optim.next(batch_size=arch_batch_size)
            self.clear_buffer()
            self.best_score_batch = None
            self.best_desc_batch = None
            for params in inputs:
                # estim step
                self.stepped(params)
            self.wait_done()
            for _, res, arch_desc in self.buffer():
                score = self.get_score(res)
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_arch_desc = arch_desc
                if self.best_score_batch is None or score > self.best_score_batch:
                    self.best_score_batch = score
                    self.best_desc_batch = arch_desc
            # save
            if config.save_arch_desc:
                self.save_arch_desc(epoch, arch_desc=self.best_desc_batch)
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4f} Best: {:.4f}'.format(epoch + 1, tot_epochs,
                                                                                self.best_score_batch or 0,
                                                                                self.best_score or 0))
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
