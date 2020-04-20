import itertools
from ..base import EstimatorBase
from ... import utils

class DefaultEstimator(EstimatorBase):
    def __init__(self, *args, save_best=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_optim = utils.get_optimizer(self.model.weights(), self.config.w_optim)
        self.lr_scheduler = utils.get_lr_scheduler(self.w_optim, self.config.lr_scheduler,
                                                   self.config.epochs)
        self.save_best = save_best

    def predict(self, ):
        pass

    def search(self, optim):
        return self.train()

    def train(self):
        metrics = self.compute_metrics()
        self.print_model_info()
        if len(metrics) > 0:
            self.logger.info('Model metrics: {}'.format(metrics))
        config = self.config
        tot_epochs = config.epochs
        self.apply_drop_path()
        best_val_top1 = 0.
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # droppath
            self.update_drop_path_prob(epoch, tot_epochs)
            # train
            trn_top1 = self.train_epoch(epoch, tot_epochs)
            # validate
            val_top1 = self.validate_epoch(epoch, tot_epochs)
            if val_top1 is None: val_top1 = trn_top1
            best_val_top1 = max(best_val_top1, val_top1)
            # save
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            if self.save_best and val_top1 >= best_val_top1:
                self.save_checkpoint(epoch, save_name='best')
        ret = {
            'best_top1': best_val_top1
        }
        ret.update(metrics)
        return ret

    def validate(self):
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0)
        return top1_avg
