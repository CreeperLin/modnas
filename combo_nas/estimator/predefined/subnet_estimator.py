import itertools
from ..base import EstimatorBase
from ... import utils
from ...core.param_space import ArchParamSpace

class SubNetEstimator(EstimatorBase):
    def __init__(self, rebuild_subnet=False, reset_subnet_params=True,
                 num_bn_batch=100, clear_subnet_bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_subnet = rebuild_subnet
        self.reset_subnet_params = reset_subnet_params
        self.num_bn_batch = num_bn_batch
        self.clear_subnet_bn = clear_subnet_bn
        self.best_top1 = 0.
        self.best_genotype = None

    def step(self, params):
        ArchParamSpace.update_params(params)
        genotype = self.model.to_genotype()
        config = self.config
        self.logger.info('Evaluating SubNet -> {}'.format(genotype))
        try:
            subnet = self.construct_subnet(genotype)
        except RuntimeError:
            ret = {'acc_top1': 0.}
            return ret
        tot_epochs = config.subnet_epochs
        if tot_epochs > 0:
            self.reset_training_states(model=subnet, tot_epochs=tot_epochs)
            self.apply_drop_path(model=subnet)
            best_val_top1 = 0.
            for epoch in itertools.count(0):
                if epoch == tot_epochs: break
                # droppath
                self.update_drop_path_prob(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
                # train
                trn_top1 = self.train_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
                # validate
                val_top1 = self.validate_epoch(epoch=epoch, tot_epochs=tot_epochs, model=subnet)
                if val_top1 is None: val_top1 = trn_top1
                best_val_top1 = max(best_val_top1, val_top1)
        else:
            best_val_top1 = self.validate_epoch(epoch=0, tot_epochs=1, model=subnet)
        metrics_result = self.compute_metrics(model=subnet)
        ret = {
            'acc_top1': best_val_top1
        }
        if not metrics_result is None:
            ret.update(metrics_result)
        if best_val_top1 > self.best_top1:
            self.best_top1 = best_val_top1
            self.best_genotype = genotype
        self.logger.info('SubNet metrics: {}'.format(ret))
        return ret

    def predict(self, ):
        pass

    def construct_subnet(self, genotype):
        config = self.config
        if self.rebuild_subnet:
            subnet = self.model_builder(genotype=genotype)
        else:
            subnet = self.model
        if self.reset_subnet_params:
            subnet.init_model(**config.get('init', {}))
        else:
            self.recompute_bn_running_statistics(num_batch=self.num_bn_batch,
                                                 model=subnet, clear=self.clear_subnet_bn)
        return subnet

    def validate(self):
        subnet = self.construct_subnet(self.model.to_genotype())
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0, model=subnet)
        return top1_avg

    def search(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        eta_m = utils.ETAMeter(tot_epochs, self.init_epoch)
        eta_m.start()
        for epoch in itertools.count(self.init_epoch+1):
            if epoch >= tot_epochs: break
            # arch step
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            batch_top1 = 0.
            for params in self.inputs:
                # estim step
                result = self.step(params)
                self.results.append(result)
                val_top1 = result['acc_top1']
                if val_top1 > batch_top1:
                    batch_top1 = val_top1
            # save
            if config.save_gt:
                self.save_genotype(epoch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            eta_m.step()
            logger.info('Search: [{:3d}/{}] Prec@1: {:.4%} Best: {:.4%} | ETA: {}'.format(
                epoch, tot_epochs, batch_top1, self.best_top1, eta_m.eta_fmt()))
        return {
            'best_top1': self.best_top1,
            'best_gt': self.best_genotype,
        }
