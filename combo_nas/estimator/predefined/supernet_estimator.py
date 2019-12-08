import torch
import torch.nn as nn
import itertools
from ..base import EstimatorBase
from ... import utils
from ...utils.profiling import tprof
from ..base import train, validate
from ...core.param_space import ArchParamSpace

class SuperNetEstimator(EstimatorBase):
    def predict(self, ):
        pass
    
    def search(self, arch_optim):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        tot_epochs = config.epochs
        model = self.model
        device = self.device

        best_top1 = 0.
        genotypes = []
        best_genotype = None
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # train
            trn_top1 = self.search_epoch(epoch, arch_optim)
            # validate
            cur_step = (epoch+1) * len(train_loader)
            val_top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config)
            genotype = model.to_genotype()
            genotypes.append(genotype)
            if val_top1 is None: val_top1 = trn_top1
            if val_top1 > best_top1:
                best_top1 = val_top1
                best_genotype = genotype
            # save
            self.save_genotype(epoch)
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)
        return best_top1, best_genotype, genotypes
    
    def search_epoch(self, epoch, arch_optim):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        lr = self.lr_scheduler.get_lr()[0]
        w_optim = self.w_optim
        model = self.model
        device = self.device
        tot_epochs = config.epochs

        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses = utils.AverageMeter()

        n_trn_batch = len(train_loader)
        cur_step = epoch*n_trn_batch
        writer.add_scalar('train/lr', lr, cur_step)
        
        if not valid_loader is None:
            val_iter = iter(valid_loader)
            n_val_batch = len(valid_loader)

        update_arch = False
        if not arch_optim is None:
            arch_epoch_start = config.arch_update_epoch_start
            arch_epoch_intv = config.arch_update_epoch_intv
            if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
                update_arch = True
                arch_update_intv = config.arch_update_intv
                if arch_update_intv == -1: # update proportionally
                    arch_update_intv = n_trn_batch // n_val_batch if not valid_loader is None else 1
                elif arch_update_intv == 0: # update every step
                    arch_update_intv = n_trn_batch
                arch_update_batch = config.arch_update_batch

        model.train()
        eta_m = utils.ETAMeter(tot_epochs, epoch, n_trn_batch)
        eta_m.start()
        tprof.timer_start('data')
        for step, (trn_X, trn_y) in enumerate(train_loader):
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            N = trn_X.size(0)
            tprof.timer_stop('data')
            # supernet step
            tprof.timer_start('train')
            w_optim.zero_grad()
            loss, logits = model.loss_logits(trn_X, trn_y, config.aux_weight)
            loss.backward()
            # gradient clipping
            if config.w_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
            w_optim.step()
            tprof.timer_stop('train')
            # arch_optim step
            if update_arch and (step+1) % arch_update_intv == 0:
                for a_batch in range(arch_update_batch):
                    if valid_loader is None:
                        tprof.timer_start('arch')
                        arch_optim.step(self)
                    else:
                        tprof.timer_start('data')
                        try:
                            val_X, val_y = next(val_iter)
                        except:
                            val_iter = iter(valid_loader)
                            val_X, val_y = next(val_iter)
                        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
                        tprof.timer_stop('data')
                        tprof.timer_start('arch')
                        arch_optim.step(self)
                    tprof.timer_stop('arch')

            prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step !=0 and step % config.print_freq == 0 or step == n_trn_batch-1:
                eta = eta_m.step(step)
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) | ETA: {eta}".format(
                        epoch+1, tot_epochs, step, n_trn_batch-1, lr, losses=losses,
                        top1=top1, top5=top5, eta=utils.format_time(eta)))

            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1
            if step < n_trn_batch-1: tprof.timer_start('data')
        logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
        tprof.print_stat('data')
        tprof.print_stat('train')
        # torch > 1.2.0
        self.lr_scheduler.step()
        return top1.avg

    def validate(self):
        config = self.config
        train_loader = self.train_loader
        valid_loader = self.valid_loader
        writer = self.writer
        logger = self.logger
        model = self.model
        device = self.device
        top1_avg = validate(valid_loader, model, writer, logger, 0, 1, 0, device, config)
        return top1_avg
