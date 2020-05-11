import torch.nn as nn
import itertools
from ..base import EstimatorBase
from ... import utils

class SuperNetEstimator(EstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_loader is None:
            self.trn_iter = None
        else:
            self.trn_iter = iter(self.train_loader)
        if self.valid_loader is None:
            self.val_iter = None
        else:
            self.val_iter = iter(self.valid_loader)
        self.cur_trn_batch = None
        self.cur_val_batch = None
        self.no_valid_warn = True
        self.best_top1 = 0.
        self.best_genotype = None
        self.reset_training_states()
        self.print_model_info()

    def predict(self, ):
        pass

    def search(self, optim):
        model = self.model
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # train
            self.model.print_arch_params(self.logger)
            trn_top1 = self.search_epoch(epoch, optim)
            # validate
            val_top1 = self.validate_epoch(epoch, tot_epochs)
            genotype = model.to_genotype()
            if val_top1 is None: val_top1 = trn_top1
            if val_top1 > self.best_top1:
                self.best_top1 = val_top1
                self.best_genotype = genotype
            self.logger.info('Search: [{:3d}/{}] Prec@1: {:.4%} -> {}'.format(epoch+1, tot_epochs, val_top1, genotype))
            # save
            if config.save_gt:
                self.save_genotype(epoch, genotype=genotype)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
            self.save_genotype(save_name='best', genotype=self.best_genotype)
        return {
            'best_top1': self.best_top1,
            'best_gt': self.best_genotype,
        }

    def get_lr(self):
        return self.lr_scheduler.get_lr()[0]

    def get_next_trn_batch(self):
        self.tprof.timer_start('data')
        try:
            trn_X, trn_y = next(self.trn_iter)
        except StopIteration:
            self.trn_iter = iter(self.train_loader)
            trn_X, trn_y = next(self.trn_iter)
        trn_X, trn_y = trn_X.to(self.device, non_blocking=True), trn_y.to(self.device, non_blocking=True)
        self.tprof.timer_stop('data')
        self.cur_trn_batch = trn_X, trn_y
        return trn_X, trn_y

    def get_cur_trn_batch(self):
        return self.cur_trn_batch

    def get_next_val_batch(self):
        if self.valid_loader is None:
            if self.no_valid_warn:
                self.logger.warning('no valid loader, returning training batch instead')
                self.no_valid_warn = False
            return self.get_next_trn_batch()
        self.tprof.timer_start('data')
        try:
            val_X, val_y = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.valid_loader)
            val_X, val_y = next(self.val_iter)
        val_X, val_y = val_X.to(self.device, non_blocking=True), val_y.to(self.device, non_blocking=True)
        self.tprof.timer_stop('data')
        self.cur_val_batch = val_X, val_y
        return val_X, val_y

    def get_cur_val_batch(self):
        return self.cur_val_batch

    def search_epoch(self, epoch, optim):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        lr = self.get_lr()
        w_optim = self.w_optim
        model = self.model
        tot_epochs = config.epochs

        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        n_trn_batch = len(train_loader)
        cur_step = epoch * n_trn_batch
        writer.add_scalar('train/lr', lr, cur_step)

        if not valid_loader is None:
            self.val_iter = iter(valid_loader)
            n_val_batch = len(valid_loader)

        update_arch = False
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
            update_arch = True
            arch_update_intv = config.arch_update_intv
            if arch_update_intv == -1: # update proportionally
                arch_update_intv = max(n_trn_batch / n_val_batch, 1) if not valid_loader is None else 1
            elif arch_update_intv == 0: # update last step
                arch_update_intv = n_trn_batch
            arch_update_batch = config.arch_update_batch

        tprof = self.tprof
        model.train()
        eta_m = utils.ETAMeter(tot_epochs * n_trn_batch, cur_step)
        eta_m.start()
        arch_step = 0
        for step in range(n_trn_batch):
            trn_X, trn_y = self.get_next_trn_batch()
            N = trn_X.size(0)
            # optim step
            if update_arch and (step+1) // arch_update_intv > arch_step:
                for _ in range(arch_update_batch):
                    tprof.timer_start('arch')
                    optim.step(self)
                    tprof.timer_stop('arch')
                arch_step += 1
            # supernet step
            tprof.timer_start('train')
            w_optim.zero_grad()
            loss, logits = self.loss_logits(trn_X, trn_y)
            loss.backward()
            # gradient clipping
            if config.w_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
            w_optim.step()
            tprof.timer_stop('train')

            prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step !=0 and step % config.print_freq == 0 or step == n_trn_batch-1:
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) | ETA: {eta}".format(
                        epoch+1, tot_epochs, step, n_trn_batch-1, lr, losses=losses,
                        top1=top1, top5=top5, eta=eta_m.eta_fmt()))

            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1
            eta_m.step()
        logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
        tprof.stat('data')
        tprof.stat('train')
        tprof.stat('arch')
        # torch > 1.2.0
        self.lr_scheduler.step()
        return top1.avg

    def validate(self):
        top1_avg = self.validate_epoch(epoch=0, tot_epochs=1, cur_step=0)
        return top1_avg
