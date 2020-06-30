import torch
import torch.nn as nn
from ...utils.optimizer import get_optimizer
from ...utils.lr_scheduler import get_lr_scheduler
from ... import utils
from ..base import TrainerBase
from .. import register_as

@register_as('ImageCls')
class ImageClsTrainer(TrainerBase):
    def __init__(self, logger=None, writer=None,
                 w_optim=None, lr_scheduler=None, expman=None,
                 w_grad_clip=0, print_freq=200):
        super().__init__(logger, writer)
        self.top1 = None
        self.top5 = None
        self.losses = None
        self.print_freq = print_freq
        self.w_grad_clip = w_grad_clip
        self.expman = expman
        self.w_optim = None
        self.lr_scheduler = None
        if isinstance(w_optim, dict):
            self.w_optim_config = w_optim
        else:
            self.w_optim = w_optim
        if isinstance(lr_scheduler, dict):
            self.lr_scheduler_config = lr_scheduler
        else:
            self.lr_scheduler = lr_scheduler
        self.reset_stats()

    def init(self, model, w_optim_config=None, lr_scheduler_config=None,
             tot_epochs=None, scale_lr=True, device=None):
        device = model.device_ids if device is None else device
        if w_optim_config is None:
            w_optim_config = self.w_optim_config
        if lr_scheduler_config is None:
            lr_scheduler_config = self.lr_scheduler_config
        if not w_optim_config is None:
            self.w_optim = get_optimizer(model.weights(), w_optim_config, device, scale_lr)
        if not lr_scheduler_config is None:
            self.lr_scheduler = get_lr_scheduler(self.w_optim, lr_scheduler_config, tot_epochs)

    def reset_stats(self):
        self.top1 = utils.AverageMeter()
        self.top5 = utils.AverageMeter()
        self.losses = utils.AverageMeter()

    def state_dict(self):
        return {
            'w_optim': self.w_optim.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        if not self.w_optim is None:
            self.w_optim.load_state_dict(sd['w_optim'])
        if not self.lr_scheduler is None:
            self.lr_scheduler.load_state_dict(sd['lr_scheduler'])

    def get_lr(self):
        return self.lr_scheduler.get_lr()[0]

    def train_epoch(self, estim, model, tot_steps, epoch, tot_epochs):
        for step in range(tot_steps):
            self.train_step(estim, model, epoch, tot_epochs, step, tot_steps)
        return {
            'acc_top1': self.top1.avg,
            'acc_top5': self.top5.avg
        }

    def train_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        cur_step = epoch * tot_steps + step
        writer = self.writer
        logger = self.logger
        w_optim = self.w_optim
        lr_scheduler = self.lr_scheduler
        lr = lr_scheduler.get_lr()[0]
        if step == 0:
            self.reset_stats()
            writer.add_scalar('train/lr', lr, cur_step)
        top1 = self.top1
        top5 = self.top5
        losses = self.losses
        print_freq = self.print_freq
        tprof = estim.tprof
        model.train()
        trn_X, trn_y = estim.get_next_trn_batch()
        N = trn_X.size(0)
        tprof.timer_start('train')
        w_optim.zero_grad()
        loss, logits = estim.loss_logits(trn_X, trn_y, model=model, mode='train')
        loss.backward()
        # gradient clipping
        if self.w_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.weights(), self.w_grad_clip)
        w_optim.step()
        tprof.timer_stop('train')
        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if print_freq != 0 and step != 0 and (step+1) % print_freq == 0 or step == tot_steps-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, tot_epochs, step+1, tot_steps, lr, losses=losses,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if step == tot_steps - 1:
            lr_scheduler.step()
            logger.info("Train: [{:3d}/{}] Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
        return loss, prec1, prec5

    def validate_epoch(self, estim, model, tot_steps, epoch=0, tot_epochs=1):
        if not tot_steps:
            return None
        for step in range(tot_steps):
            self.validate_step(estim, model, epoch, tot_epochs, step, tot_steps)
        return {
            'acc_top1': self.top1.avg,
            'acc_top5': self.top5.avg
        }

    def validate_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        if step == 0:
            self.reset_stats()
        cur_step = epoch * tot_steps + step
        writer = self.writer
        logger = self.logger
        top1 = self.top1
        top5 = self.top5
        losses = self.losses
        print_freq = self.print_freq
        model.eval()
        tprof = estim.tprof
        tprof.timer_start('validate')
        with torch.no_grad():
            val_X, val_y = estim.get_next_val_batch()
            loss, logits = estim.loss_logits(val_X, val_y, model=model, mode='eval')
        tprof.timer_stop('validate')
        prec1, prec5 = utils.accuracy(logits, val_y, topk=(1, 5))
        N = val_X.size(0)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if print_freq != 0 and ((step+1) % print_freq == 0 or step+1 == tot_steps):
            logger.info(
                "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, tot_epochs, step+1, tot_steps, losses=losses,
                    top1=top1, top5=top5))
        if step+1 == tot_steps:
            writer.add_scalar('val/loss', losses.avg, cur_step)
            writer.add_scalar('val/top1', top1.avg, cur_step)
            writer.add_scalar('val/top5', top5.avg, cur_step)
            logger.info("Valid: [{:3d}/{}] Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
        return top1.avg
