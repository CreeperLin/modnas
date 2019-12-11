import torch
import torch.nn as nn
import itertools
from ..base import EstimatorBase
from ... import utils
from ...utils.profiling import tprof
from ...core.param_space import ArchParamSpace

class SubNetEstimator(EstimatorBase):
    def step(self):
        config = self.config
        tot_epochs = config.subnet_epochs
        subnet = self.construct_subnet()
        w_optim = utils.get_optim(subnet.weights(), config.w_optim)
        lr_scheduler = utils.get_lr_scheduler(w_optim, config.lr_scheduler, tot_epochs)
        # train
        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # train
            trn_top1 = self.train_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet,
                                        w_optim=w_optim, lr_scheduler=lr_scheduler)
            # validate
            val_top1 = self.validate_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
            if val_top1 is None: val_top1 = trn_top1
            best_val_top1 = max(best_val_top1, val_top1)
        return best_val_top1

    def predict(self, ):
        pass
    
    def construct_subnet(self):
        config = self.config
        # supernet based
        self.model.init_model(config.init)
        return self.model
        # subnet based
        # convert_fn = None
        # net = build_arch_space(config.model.type, config.model)
        # drop_path = 0.0
        # genotype = self.model.to_genotype()
        # model = convert_from_genotype(net, genotype, convert_fn, drop_path)
        # model = NASController(model, self.model.criterion, dev_list=None)
        # model.init_model(config.init)
    
    def train(self):
        config = self.config
        tot_epochs = config.epochs
        subnet = self.construct_subnet()

        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # train
            trn_top1 = self.train_epoch(epoch=epoch, tot_epochs=tot_epochs)
            # validate
            val_top1 = self.validate_epoch(epoch=epoch, tot_epochs=tot_epochs)
            if val_top1 is None: val_top1 = trn_top1
            best_val_top1 = max(best_val_top1, val_top1)
            # save
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
        return best_val_top1

    def validate(self):
        subnet = self.construct_subnet()
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0, model=subnet)
        return top1_avg

    def search(self, arch_optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs

        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        best_top1 = 0.
        best_genotype = None
        genotypes = []
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                arch_optim.step(self)
            # estim step
            genotype = self.model.to_genotype()
            logger.info('Evaluating SubNet genotype = {}'.format(genotype))
            val_top1 = self.step()
            if val_top1 > best_top1:
                best_top1 = val_top1
                best_genotype = genotype
            # save
            self.save_genotype(epoch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            logger.info('SubNet Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%}'.format(epoch, tot_epochs, val_top1, best_top1))
        return best_top1, best_genotype, genotypes