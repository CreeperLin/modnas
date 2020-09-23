import itertools
import traceback
from ..base import EstimBase
from ... import utils
from ...core.param_space import ArchParamSpace
from .. import register


@register
class SubNetEstim(EstimBase):
    def __init__(self, rebuild_subnet=False, num_bn_batch=100, clear_subnet_bn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_subnet = rebuild_subnet
        self.num_bn_batch = num_bn_batch
        self.clear_subnet_bn = clear_subnet_bn
        self.best_score = None
        self.best_arch_desc = None

    def step(self, params):
        ArchParamSpace.update_params(params)
        arch_desc = self.exporter(self.model)
        config = self.config
        try:
            self.construct_subnet(arch_desc)
        except RuntimeError:
            self.logger.info('subnet construct failed:\n{}'.format(traceback.format_exc()))
            ret = {'error_no': -1}
            return ret
        tot_epochs = config.subnet_epochs
        if tot_epochs > 0:
            self.reset_training_states(tot_epochs=tot_epochs)
            for epoch in itertools.count(0):
                if epoch == tot_epochs:
                    break
                # train
                self.train_epoch(epoch=epoch, tot_epochs=tot_epochs)
        ret = self.compute_metrics()
        best_score = self.get_score(ret)
        if self.best_score is None or (best_score is not None and best_score > self.best_score):
            self.best_score = best_score
            self.best_arch_desc = arch_desc
        self.logger.info('Evaluate: {} -> {}'.format(arch_desc, ret))
        return ret

    def construct_subnet(self, arch_desc):
        if self.rebuild_subnet:
            self.model = self.constructor(arch_desc=arch_desc)
        else:
            utils.recompute_bn_running_statistics(self.model, self.trainer, self.num_bn_batch, self.clear_subnet_bn)

    def valid(self):
        self.construct_subnet(self.exporter(self.model))
        return self.valid_epoch(epoch=0, tot_epochs=1)

    def run(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        eta_m = utils.ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch in itertools.count(self.cur_epoch + 1):
            if epoch >= tot_epochs:
                break
            # arch step
            if not optim.has_next():
                logger.info('Search: finished')
                break
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                optim.step(self)
            self.inputs = optim.next(batch_size=arch_batch_size)
            self.results = []
            batch_best = None
            for params in self.inputs:
                # estim step
                result = self.step(params)
                self.results.append(result)
                val_score = self.get_score(result)
                if batch_best is None or val_score > batch_best:
                    batch_best = val_score
            # save
            if config.save_gt:
                self.save_arch_desc(epoch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            self.save_arch_desc(save_name='best', arch_desc=self.best_arch_desc)
            eta_m.step()
            logger.info('Search: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                epoch + 1, tot_epochs, batch_best or 0, self.best_score or 0, eta_m.eta_fmt()))
        return {
            'best_score': self.best_score,
            'best_arch': self.best_arch_desc,
        }
