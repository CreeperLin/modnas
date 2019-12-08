# import logging
import torch
import torch.nn as nn
from .. import utils
from ..utils.profiling import tprof
from ..core.nas_modules import NASModule
from ..arch_space import genotypes as gt

def train(train_loader, model, writer, logger, w_optim, lr_scheduler, epoch, tot_epochs, device, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    n_trn_batch = len(train_loader)
    cur_step = epoch*n_trn_batch
    lr = lr_scheduler.get_lr()[0]
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    eta_m = utils.ETAMeter(tot_epochs, epoch, n_trn_batch)
    eta_m.start()
    tprof.timer_start('data')
    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)
        tprof.timer_stop('data')
        tprof.timer_start('train')
        w_optim.zero_grad()
        loss, logits = model.loss_logits(trn_X, trn_y, config.aux_weight)
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
    lr_scheduler.step()
    return top1.avg


def validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config):
    if valid_loader is None: return None
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    n_val_batch = len(valid_loader)
    model.eval()
    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_loader):
            val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
            N = val_X.size(0)

            tprof.timer_start('validate')
            loss, logits = model.loss_logits(val_X, val_y, config.aux_weight)
            tprof.timer_stop('validate')

            prec1, prec5 = utils.accuracy(logits, val_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step !=0 and step % config.print_freq == 0 or step == n_val_batch-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, tot_epochs, step, n_val_batch-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.print_stat('validate')

    return top1.avg

def save_checkpoint(expman, model, w_optim, lr_scheduler, epoch, logger):
    try:
        save_path = expman.join('chkpt', 'chkpt_{:03d}.pt'.format(epoch+1))
        torch.save({
            'model': model.state_dict(),
            # 'arch': NASModule.nasmod_state_dict(),
            'w_optim': w_optim.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, save_path)
        logger.info("Saved checkpoint to: %s" % save_path)
    except Exception as e:
        logger.error("Save checkpoint failed: "+str(e))

def save_genotype(expman, genotype, epoch, logger):
    try:
        logger.info("genotype = {}".format(genotype))
        save_path = expman.join('output', 'gene_{:03d}.gt'.format(epoch+1))
        gt.to_file(genotype, save_path)
        logger.info("Saved genotype to: %s" % save_path)
    except Exception as e:
        logger.error("Save genotype failed: "+str(e))

class EstimatorBase():
    def __init__(self, config, expman, train_loader, valid_loader, 
                model, writer, logger, device):
        self.config = config
        self.expman = expman
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.writer = writer
        self.logger = logger
        self.device = device
        self.init_epoch = -1

        self.w_optim = utils.get_optim(self.model.weights(), config.w_optim)
        self.lr_scheduler = utils.get_lr_scheduler(self.w_optim, config.lr_scheduler, config.epochs)
    
    def predict(self, ):
        pass

    def train(self):
        pass
    
    def validate(self, ):
        pass
    
    def search(self, arch_optim):
        pass
    
    def save(self, epoch):
        self.save_genotype(epoch)
        self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        expman = self.expman
        model = self.model
        w_optim = self.w_optim
        lr_scheduler = self.lr_scheduler
        logger = self.logger
        save_checkpoint(expman, model, w_optim, lr_scheduler, epoch, logger)
    
    def save_genotype(self, epoch):
        expman = self.expman
        genotype = self.model.to_genotype()
        logger = self.logger
        save_genotype(expman, genotype, epoch, logger)
    
    def load(self, chkpt_path):
        if chkpt_path is None:
            self.logger.info("Estimator: Starting new run")
            return
        self.logger.info("Estimator: Resuming from checkpoint: {}".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.w_optim.load_state_dict(checkpoint['w_optim'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.init_epoch = checkpoint['epoch']