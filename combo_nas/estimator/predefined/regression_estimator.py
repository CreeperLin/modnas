import itertools
from ..base import EstimatorBase
from ... import utils
from ...utils.profiling import tprof
from ..base import train
from ...core.param_space import ArchParamSpace

class ArchPredictor():
    def __init__(self):
        pass
    
    def fit(self, ):
        pass
    
    def predict(self, ):
        pass


class RegressionEstimator(EstimatorBase):
    def step(self):
        predictor = self.predictor
        model = self.model
        genotype = model.to_genotype()
        best_val_top1 = predictor.predict(genotype)
        return genotype, best_val_top1

    def predict(self, ):
        pass
    
    def train(self):
        config = self.config
        train_loader = self.train_loader
        writer = self.writer
        logger = self.logger
        tot_epochs = config.epochs
        lr = self.lr_scheduler.get_lr()[0]
        device = self.device
        model = self.model

        best_val_top1 = 0.
        for epoch in itertools.count(init_epoch+1):
            if epoch == tot_epochs: break
            cur_step = (epoch+1) * len(train_loader)
            # train
            trn_top1 = train(train_loader, model, writer, logger, w_optim, lr_scheduler, epoch, tot_epochs, device, config)
            # validate
            val_top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config)
            if val_top1 is None: val_top1 = trn_top1
            best_val_top1 = max(best_val_top1, val_top1)
            # save
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save()
        return best_val_top1

    def validate(self):
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0)
        return top1_avg

    def search(self, arch_optim):
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
                arch_optim.step(self)
            next_batch = arch_optim.next(batch_size=arch_batch_size)
            best_top1_batch = 0.
            best_gt_batch = None
            for params in next_batch:
                ArchParamSpace.set_params_map(params)
                # estim step
                genotype, val_top1 = self.step()
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    best_genotype = genotype
                if val_top1 > best_top1_batch:
                    best_top1_batch = val_top1
                    best_gt_batch = genotype
            genotypes.append(best_gt_batch)
            # save
            self.save_genotype(epoch, genotype=best_gt_batch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            logger.info('Regression Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%}'.format(epoch, tot_epochs, best_top1_batch, best_top1))
        return best_top1, best_genotype, genotypes