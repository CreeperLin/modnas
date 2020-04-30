import itertools
from ..base import EstimatorBase
from ...core.param_space import ArchParamSpace

class ArchPredictor():
    def __init__(self):
        pass

    def fit(self, ):
        pass

    def predict(self, genotype):
        pass


class RegressionEstimator(EstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = None
        self.best_score = 0.
        self.best_genotype = None

    def step(self, params):
        ArchParamSpace.update_params(params)
        predictor = self.predictor
        model = self.model
        if model is None:
            genotype = list(params.values())
            score = predictor.predict(params)
        else:
            genotype = model.to_genotype()
            score = predictor.predict(genotype)
        if score > self.best_score:
            self.best_score = score
            self.best_genotype = genotype
        return genotype, score

    def predict(self, ):
        pass

    def train(self):
        pass

    def validate(self):
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0)
        return top1_avg

    def search(self, optim):
        config = self.config
        tot_epochs = config.epochs
        logger = self.logger
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            best_score_batch = 0.
            best_gt_batch = None
            for params in self.inputs:
                # estim step
                genotype, score = self.step(params)
                if score > best_score_batch:
                    best_score_batch = score
                    best_gt_batch = genotype
                self.results.append(score)
            # save
            if config.save_gt:
                self.save_genotype(epoch, genotype=best_gt_batch)
            self.save_genotype(save_name='best', genotype=self.best_genotype)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%}'.format(
                epoch, tot_epochs, best_score_batch, self.best_score))
        return {
            'best_score': self.best_score,
            'best_gt': self.best_genotype,
        }
