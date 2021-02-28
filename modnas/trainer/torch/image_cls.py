from collections import OrderedDict
import torch
import torch.nn as nn
from ... import backend
from ...utils import AverageMeter
from ..base import TrainerBase
from modnas.registry.trainer import register


def accuracy(output, target, topk=(1, )):
    """Compute the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


@register
class ImageClsTrainer(TrainerBase):
    def __init__(self,
                 writer=None,
                 expman=None,
                 device='cuda',
                 data_provider=None,
                 optimizer=None,
                 lr_scheduler=None,
                 criterion='CrossEntropyLoss',
                 w_grad_clip=0,
                 print_freq=200):
        super().__init__(writer)
        self.device = device
        self.top1 = None
        self.top5 = None
        self.losses = None
        self.print_freq = print_freq
        self.w_grad_clip = w_grad_clip
        self.expman = expman
        self.optimizer = None
        self.lr_scheduler = None
        self.data_provider = None
        self.criterion = None
        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'data_provider': data_provider,
            'criterion': criterion,
        }
        self.config = config
        self.reset_stats()

    def init(self, model, config=None):
        self.config.update(config or {})
        if self.config['optimizer']:
            self.optimizer = backend.get_optimizer(model.parameters(), self.config['optimizer'], config)
        if self.config['lr_scheduler']:
            self.lr_scheduler = backend.get_lr_scheduler(self.optimizer, self.config['lr_scheduler'], config)
        if self.config['data_provider']:
            self.data_provider = backend.get_data_provider(self.config['data_provider'])
        if self.config['criterion']:
            self.criterion = backend.get_criterion(self.config['criterion'], getattr(model, 'device_ids', None))
        self.device = self.config.get('device', self.device)

    def get_num_train_batch(self, epoch):
        return 0 if self.data_provider is None else self.data_provider.get_num_train_batch(epoch=epoch)

    def get_num_valid_batch(self, epoch):
        return 0 if self.data_provider is None else self.data_provider.get_num_valid_batch(epoch=epoch)

    def get_next_train_batch(self):
        return self.proc_batch(self.data_provider.get_next_train_batch())

    def get_next_valid_batch(self):
        return self.proc_batch(self.data_provider.get_next_valid_batch())

    def proc_batch(self, batch):
        return tuple(v.to(device=self.device, non_blocking=True) for v in batch)

    def reset_stats(self):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        if self.optimizer is not None:
            self.optimizer.load_state_dict(sd['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(sd['lr_scheduler'])

    def get_lr(self):
        if self.lr_scheduler:
            if hasattr(self.lr_scheduler, 'get_last_lr'):
                return self.lr_scheduler.get_last_lr()[0]
            return self.lr_scheduler.get_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    def get_optimizer(self):
        return self.optimizer

    def loss(self, output=None, data=None, model=None):
        return None if self.criterion is None else self.criterion(None, None, output, *data)

    def train_epoch(self, estim, model, tot_steps, epoch, tot_epochs):
        for step in range(tot_steps):
            self.train_step(estim, model, epoch, tot_epochs, step, tot_steps)
        return {
            'acc_top1': self.top1.avg,
            'acc_top5': self.top5.avg,
            'loss': self.losses.avg,
        }

    def train_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        cur_step = epoch * tot_steps + step
        writer = self.writer
        logger = self.logger
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        lr = self.get_lr()
        if step == 0:
            self.data_provider.reset_train_iter()
            self.reset_stats()
            writer.add_scalar('train/lr', lr, cur_step)
        top1 = self.top1
        top5 = self.top5
        losses = self.losses
        print_freq = self.print_freq
        model.train()
        batch = self.get_next_train_batch()
        trn_X, trn_y = batch
        N = trn_X.size(0)
        optimizer.zero_grad()
        loss, logits = estim.loss_output(batch, model=model, mode='train')
        loss.backward()
        # gradient clipping
        if self.w_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.weights(), self.w_grad_clip)
        optimizer.step()
        prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if print_freq != 0 and ((step + 1) % print_freq == 0 or step + 1 == tot_steps):
            logger.info('Train: [{:3d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {:.3f} Prec@(1,5) ({:.1%}, {:.1%})'.format(
                epoch + 1, tot_epochs, step + 1, tot_steps, lr, losses.avg, top1.avg, top5.avg))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        if step == tot_steps - 1:
            lr_scheduler.step()
            logger.info("Train: [{:3d}/{}] Prec@1 {:.4%}".format(epoch + 1, tot_epochs, top1.avg))
        return loss, prec1, prec5

    def valid_epoch(self, estim, model, tot_steps, epoch=0, tot_epochs=1):
        if not tot_steps:
            return None
        for step in range(tot_steps):
            self.valid_step(estim, model, epoch, tot_epochs, step, tot_steps)
        return OrderedDict([
            ('acc_top1', self.top1.avg),
            ('acc_top5', self.top5.avg),
            ('loss', self.losses.avg),
        ])

    def valid_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        if step == 0:
            self.data_provider.reset_valid_iter()
            self.reset_stats()
        cur_step = epoch * tot_steps + step
        writer = self.writer
        logger = self.logger
        top1 = self.top1
        top5 = self.top5
        losses = self.losses
        print_freq = self.print_freq
        model.eval()
        with torch.no_grad():
            batch = self.get_next_valid_batch()
            val_X, val_y = batch
            loss, logits = estim.loss_output(batch, model=model, mode='eval')
        prec1, prec5 = accuracy(logits, val_y, topk=(1, 5))
        N = val_X.size(0)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
        if print_freq != 0 and ((step + 1) % print_freq == 0 or step + 1 == tot_steps):
            logger.info('Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Prec@(1,5) ({:.1%}, {:.1%})'.format(
                epoch + 1, tot_epochs, step + 1, tot_steps, losses.avg, top1.avg, top5.avg))
        if step + 1 == tot_steps:
            writer.add_scalar('val/loss', losses.avg, cur_step)
            writer.add_scalar('val/top1', top1.avg, cur_step)
            writer.add_scalar('val/top5', top5.avg, cur_step)
            logger.info("Valid: [{:3d}/{}] Prec@1 {:.4%}".format(epoch + 1, tot_epochs, top1.avg))
        return top1.avg
