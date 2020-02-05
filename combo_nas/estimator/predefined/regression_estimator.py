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

    def step(self):
        predictor = self.predictor
        model = self.model
        genotype = model.to_genotype()
        best_val_top1 = predictor.predict(genotype)
        return genotype, best_val_top1

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
        arch_batch_size = config.get('arch_update_batch', 1)
        best_top1 = 0.
        best_genotype = None
        genotypes = []
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            best_top1_batch = 0.
            best_gt_batch = None
            for params in self.inputs:
                ArchParamSpace.set_params_map(params)
                # estim step
                genotype, val_top1 = self.step()
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    best_genotype = genotype
                if val_top1 > best_top1_batch:
                    best_top1_batch = val_top1
                    best_gt_batch = genotype
                self.results.append(val_top1)
            genotypes.append(best_gt_batch)
            # save
            if config.save_gt:
                self.save_genotype(epoch, genotype=best_gt_batch)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%}'.format(
                epoch, tot_epochs, best_top1_batch, best_top1))
        return {
            'best_top1': best_top1,
            'best_gt': best_genotype,
            'gts': genotypes
        }
