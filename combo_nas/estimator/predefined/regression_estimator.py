import torch
import torch.nn as nn
import itertools
from ..base import EstimatorBase
from ... import utils
from ...utils.profiling import tprof
from ..base import train
from ...core.param_space import ArchParamSpace

class RegressionEstimator(EstimatorBase):
    def step(self):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        lr = self.lr_scheduler.get_lr()[0]
        model = self.model
        device = self.device
        tot_epochs = config.epochs
        # train
        genotype = model.to_genotype()
        best_val_top1 = self.predictor.predict(genotype)
        return best_val_top1

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

        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                arch_optim.step(self)
            # estim step
            self.step()
            # save
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save()
        return best_top1, best_genotype, genotypes