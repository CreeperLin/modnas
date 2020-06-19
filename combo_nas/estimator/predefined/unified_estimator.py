import itertools
from ..base import EstimatorBase
from ... import utils
from ...core.param_space import ArchParamSpace

class UnifiedEstimator(EstimatorBase):
    def __init__(self, train_epochs=1, train_steps=-1, reset_training=False,
                 eval_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if train_steps != 0:
            train_epochs = 1
            eval_steps = self.n_trn_batch
            if train_steps == -1:
                if self.n_val_batch == 0:
                    raise ValueError('argument train_steps required')
                train_steps = max(round(self.n_trn_batch / self.n_val_batch), 1)
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.reset_training = reset_training
        self.eval_steps = eval_steps
        self.cur_step = -1
        self.best_score = None
        self.best_genotype = None
        self.reset_training_states()
        self.print_model_info()

    def step(self, params):
        ArchParamSpace.update_params(params)
        n_trn_batch = self.n_trn_batch
        train_epochs = self.train_epochs
        train_steps = self.train_steps or n_trn_batch
        if self.reset_training:
            self.reset_training_states(tot_epochs=train_epochs)
        for epoch in range(train_epochs):
            for _ in range(train_steps):
                self.cur_step += 1
                if self.cur_step >= n_trn_batch:
                    self.cur_step = -1
                    break
                self.train_step(model=self.model,
                                epoch=epoch, tot_epochs=train_epochs,
                                step=self.cur_step, tot_steps=n_trn_batch)
        if (self.cur_step+1) % self.eval_steps != 0:
            return {'default': None}
        genotype = self.model.to_genotype()
        ret = self.compute_metrics()
        self.logger.info('Evaluate: {} -> {}'.format(genotype, ret))
        score = self.get_score(ret)
        if self.best_score is None or not score is None and score > self.best_score:
            self.best_score = score
            self.best_genotype = genotype
        return ret

    def search(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        train_steps = self.train_steps
        n_epoch_steps = 1 if train_steps == 0 else (self.n_trn_batch + train_steps - 1) // train_steps
        self.cur_epoch += 1
        eta_m = utils.ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch_step in itertools.count(0):
            epoch = self.cur_epoch
            if epoch >= tot_epochs: break
            # arch step
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
            if (epoch_step+1) % n_epoch_steps != 0:
                continue
            # save
            if config.save_gt:
                self.save_genotype()
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint()
            self.save_genotype(save_name='best', genotype=self.best_genotype)
            eta_m.step()
            logger.info('Search: [{:3d}/{}] Current: {} Best: {} | ETA: {}'.format(
                self.cur_epoch+1, tot_epochs, batch_best, self.best_score, eta_m.eta_fmt()))
            self.cur_epoch += 1
        return {
            'best_score': self.best_score,
            'best_gt': self.best_genotype,
        }
