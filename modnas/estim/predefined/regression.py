import itertools
from ..base import EstimBase
from ...core.param_space import ParamSpace
from .. import register


@register
class RegressionEstim(EstimBase):
    def __init__(self, *args, predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.best_score = 0.
        self.best_arch_desc = None

    def step(self, params):
        ParamSpace().update_params(params)
        arch_desc = self.get_arch_desc()
        score = self.predictor.predict(arch_desc)
        if score > self.best_score:
            self.best_score = score
            self.best_arch_desc = arch_desc
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
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            best_score_batch = 0.
            best_gt_batch = None
            for params in self.inputs:
                # estim step
                score = self.step(params)
                if score > best_score_batch:
                    best_score_batch = score
                    best_gt_batch = self.get_arch_desc()
                self.results.append(score)
            # save
            if config.save_arch_desc:
                self.save_arch_desc(epoch, arch_desc=best_gt_batch)
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4f} Best: {:.4f}'.format(epoch + 1, tot_epochs, best_score_batch or 0,
                                                                                self.best_score or 0))
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
