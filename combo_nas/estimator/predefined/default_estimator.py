import torch
import torch.nn as nn
import itertools
from ..base import EstimatorBase
from ... import utils
from ...utils.profiling import tprof
from ..base import train, validate

class DefaultEstimator(EstimatorBase):
    def predict(self, ):
        pass
    
    def search(self, arch_optim):
        return self.train()
    
    def train(self):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        tot_epochs = config.epochs
        lr_scheduler = self.lr_scheduler
        w_optim = self.w_optim
        device = self.device
        model = self.model

        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
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
                self.save_checkpoint(epoch)
        return best_val_top1

    def validate(self):
        config = self.config
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        device = self.device
        model = self.model
        top1_avg = validate(valid_loader, model, writer, logger, 0, 1, 0, device, config)
        return top1_avg