import itertools
from ..base import EstimatorBase
from ...arch_space.droppath import update_drop_path_prob, apply_drop_path
from ...utils import ETAMeter

class DefaultEstimator(EstimatorBase):
    def __init__(self, *args, save_best=True, valid_intv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_best = save_best
        self.best_score = None
        self.valid_intv = valid_intv

    def predict(self, ):
        pass

    def run(self, optim):
        return self.train()

    def train(self):
        self.reset_training_states()
        self.print_model_info()
        config = self.config
        tot_epochs = config.epochs
        drop_prob = self.config.drop_path_prob
        if drop_prob > 0:
            apply_drop_path(self.model)
        eta_m = ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            # droppath
            if drop_prob > 0:
                update_drop_path_prob(self.model, drop_prob, epoch, tot_epochs)
            # train
            trn_res = self.train_epoch(epoch, tot_epochs)
            # validate
            if epoch+1 == tot_epochs or (not self.valid_intv is None and not (epoch+1) % self.valid_intv):
                val_res = self.validate_epoch(epoch, tot_epochs)
                if val_res is None: val_res = trn_res
                val_score = self.get_score(val_res)
            else:
                val_score = None
            # save
            if not val_score is None and (self.best_score is None or val_score > self.best_score):
                self.best_score = val_score
                if self.save_best:
                    self.save_checkpoint(epoch, save_name='best')
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            eta_m.step()
            self.logger.info('Default: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                epoch+1, tot_epochs, val_score or 0, self.best_score or 0, eta_m.eta_fmt()))
        metrics = self.compute_metrics()
        ret = {
            'best_score': self.best_score
        }
        ret.update(metrics)
        return ret
