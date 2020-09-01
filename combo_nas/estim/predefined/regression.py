import itertools
from ..base import EstimatorBase
from ...core.param_space import ArchParamSpace


class ArchPredictor():
    def __init__(self):
        pass

    def fit(self, ):
        pass

    def predict(self, arch_desc):
        pass


class RegressionEstimator(EstimatorBase):
    def __init__(self, *args, predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.best_score = 0.
        self.best_arch_desc = None

    def step(self, params):
        ArchParamSpace.update_params(params)
        predictor = self.predictor
        arch_desc = self.exporter(self.model)
        score = predictor.predict(arch_desc)
        if score > self.best_score:
            self.best_score = score
            self.best_arch_desc = arch_desc
        return arch_desc, score

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
                arch_desc, score = self.step(params)
                if score > best_score_batch:
                    best_score_batch = score
                    best_gt_batch = arch_desc
                self.results.append(score)
            # save
            if config.save_gt:
                self.save_arch_desc(epoch, arch_desc=best_gt_batch)
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4f} Best: {:.4f}'.format(epoch + 1, tot_epochs, best_score_batch or 0,
                                                                                self.best_score or 0))
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
