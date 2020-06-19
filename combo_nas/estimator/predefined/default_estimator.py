import itertools
from ..base import EstimatorBase
from ...arch_space.droppath import update_drop_path_prob, apply_drop_path

class DefaultEstimator(EstimatorBase):
    def __init__(self, *args, save_best=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_training_states()
        self.save_best = save_best
        self.best_score = None
        self.print_model_info()

    def predict(self, ):
        pass

    def search(self, optim):
        return self.train()

    def train(self):
        config = self.config
        tot_epochs = config.epochs
        drop_prob = self.config.drop_path_prob
        if drop_prob > 0:
            apply_drop_path(self.model)
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            # droppath
            if drop_prob > 0:
                update_drop_path_prob(self.model, drop_prob, epoch, tot_epochs)
            # train
            trn_res = self.train_epoch(epoch, tot_epochs)
            # validate
            val_res = self.validate_epoch(epoch, tot_epochs)
            if val_res is None: val_res = trn_res
            val_score = self.get_score(val_res)
            # save
            if self.best_score is None or val_score > self.best_score:
                self.best_score = val_score
                if self.save_best:
                    self.save_checkpoint(epoch, save_name='best')
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
        metrics = self.compute_metrics()
        ret = {
            'best_score': self.best_score
        }
        ret.update(metrics)
        return ret
