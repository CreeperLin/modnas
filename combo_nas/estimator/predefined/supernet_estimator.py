import itertools
from ..base import EstimatorBase
from ... import utils

class SuperNetEstimator(EstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_score = None
        self.best_genotype = None
        self.reset_training_states()

    def search(self, optim):
        model = self.model
        config = self.config
        tot_epochs = config.epochs
        eta_m = utils.ETAMeter(tot_epochs, self.cur_epoch)
        eta_m.start()
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            # train
            self.model.print_arch_params(self.logger)
            self.search_epoch(epoch, optim)
            # eval
            genotype = model.to_genotype()
            mt_ret = self.compute_metrics()
            self.logger.info('Evaluate: {} -> {}'.format(genotype, mt_ret))
            score = self.get_score(mt_ret)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_genotype = genotype
            # save
            if config.save_gt:
                self.save_genotype(epoch, genotype=genotype)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            self.save_genotype(save_name='best', genotype=self.best_genotype)
            eta_m.step()
            self.logger.info('Search: [{:3d}/{}] Current: {} Best: {} | ETA: {}'.format(
                epoch+1, tot_epochs, score, self.best_score, eta_m.eta_fmt()))
        return {
            'best_score': self.best_score,
            'best_gt': self.best_genotype,
        }

    def search_epoch(self, epoch, optim):
        config = self.config
        n_trn_batch = self.n_trn_batch
        n_val_batch = self.n_val_batch
        tot_epochs = config.epochs
        update_arch = False
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
            update_arch = True
            arch_update_intv = config.arch_update_intv
            if arch_update_intv == -1: # update proportionally
                arch_update_intv = max(n_trn_batch / n_val_batch, 1) if n_val_batch else 1
            elif arch_update_intv == 0: # update last step
                arch_update_intv = n_trn_batch
            arch_update_batch = config.arch_update_batch
        tprof = self.tprof
        arch_step = 0
        for step in range(n_trn_batch):
            # optim step
            if update_arch and (step+1) // arch_update_intv > arch_step:
                for _ in range(arch_update_batch):
                    tprof.timer_start('arch')
                    optim.step(self)
                    tprof.timer_stop('arch')
                arch_step += 1
            # supernet step
            optim.next()
            self.trainer.train_step(estim=self, model=self.model,
                                    epoch=epoch, tot_epochs=tot_epochs,
                                    step=step, tot_steps=n_trn_batch)
        tprof.stat('data')
        tprof.stat('train')
        tprof.stat('arch')
