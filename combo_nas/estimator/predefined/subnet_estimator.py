import itertools
from ..base import EstimatorBase
from ... import utils
from ...core.param_space import ArchParamSpace

class SubNetEstimator(EstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, params):
        ArchParamSpace.update_params(params)
        config = self.config
        tot_epochs = config.subnet_epochs
        subnet = self.construct_subnet()
        self.print_model_info(model=subnet)
        w_optim = utils.get_optim(subnet.weights(), config.w_optim)
        lr_scheduler = utils.get_lr_scheduler(w_optim, config.lr_scheduler, tot_epochs)
        # train subnet
        self.apply_drop_path(model=subnet)
        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # droppath
            self.update_drop_path_prob(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
            # train
            trn_top1 = self.train_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet,
                                        w_optim=w_optim, lr_scheduler=lr_scheduler)
            # validate
            val_top1 = self.validate_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
            if val_top1 is None: val_top1 = trn_top1
            best_val_top1 = max(best_val_top1, val_top1)
        metrics_result = self.compute_metrics(subnet)
        ret = {
            'best_top1': best_val_top1
        }
        if not metrics_result is None:
            ret.update(metrics_result)
        return ret

    def predict(self, ):
        pass

    def construct_subnet(self):
        config = self.config
        # supernet based
        self.model.init_model(config.init)
        return self.model

    def validate(self):
        subnet = self.construct_subnet()
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0, model=subnet)
        return top1_avg

    def search(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs

        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.get('arch_update_batch', 1)
        best_top1 = 0.
        best_genotype = None
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            batch_top1 = 0.
            for params in self.inputs:
                # estim step
                genotype = self.model.to_genotype()
                logger.info('Evaluating SubNet genotype = {}'.format(genotype))
                result = self.step(params)
                self.results.append(result)
                val_top1 = result['best_top1']
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    best_genotype = genotype
                if val_top1 > batch_top1:
                    batch_top1 = val_top1
            # save
            if config.save_gt:
                self.save_genotype(epoch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%}'.format(epoch, tot_epochs, batch_top1, best_top1))
        return {
            'best_top1': best_top1,
            'best_gt': best_genotype,
        }
