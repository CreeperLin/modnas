# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .. import utils
from .visualize import plot
from .profiling import tprof
from ..arch_space import genotypes as gt
from ..core.nas_modules import NASModule

def save_checkpoint(expman, model, w_optim, arch_optim, lr_scheduler, epoch, logger):
    try:
        save_path = expman.join('chkpt', 'chkpt_{:03d}.pt'.format(epoch+1))
        torch.save({
            'model': model.state_dict(),
            'arch': NASModule.nasmod_state_dict(),
            'w_optim': w_optim.state_dict(),
            'arch_optim': None if arch_optim is None else arch_optim.state_dict(),
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
    except:
        logger.error("Save genotype failed")


def search(config, chkpt_path, expman, train_loader, valid_loader, model, arch_optim, writer, logger, device):
    w_optim = utils.get_optim(model.weights(), config.w_optim)
    lr_scheduler = utils.get_lr_scheduler(w_optim, config.lr_scheduler, config.epochs)
    
    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: {}".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        NASModule.nasmod_load_state_dict(checkpoint['arch'])
        w_optim.load_state_dict(checkpoint['w_optim'])
        arch_optim.load_state_dict(checkpoint['arch_optim'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']
        resume = True
    else:
        logger.info("Starting new training run")
        init_epoch = -1
        resume = False
    
    logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))

    # warmup training loop
    logger.info('begin warmup training')
    try:
        if not resume and config.warmup_epochs > 0:
            warmup_lr_scheduler = utils.get_lr_scheduler(w_optim, config.lr_scheduler, config.warmup_epochs)
            tot_epochs = config.warmup_epochs
            for epoch in itertools.count(init_epoch+1):
                if epoch == tot_epochs: break

                lr = warmup_lr_scheduler.get_lr()[0]

                # training
                train(train_loader, None, model, writer, logger, None, w_optim, lr, epoch, tot_epochs, device, config)

                # validation
                cur_step = (epoch+1) * len(train_loader)
                top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config)

                warmup_lr_scheduler.step()
    except KeyboardInterrupt:
        logger.info('skipped')
    
    save_checkpoint(expman, model, w_optim, arch_optim, lr_scheduler, init_epoch, logger)
    save_genotype(expman, model.to_genotype(), init_epoch, logger)

    # training loop
    logger.info('begin w/a training')
    best_top1 = 0.
    tot_epochs = config.epochs
    genotypes = []
    for epoch in itertools.count(init_epoch+1):
        if epoch == tot_epochs: break
        lr = lr_scheduler.get_lr()[0]
        model.print_alphas(logger)
        # training
        trn_top1 = train(train_loader, valid_loader, model, writer, logger, arch_optim, w_optim, lr, epoch, tot_epochs, device, config)
        # validation
        cur_step = (epoch+1) * len(train_loader)
        val_top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config) 
        if val_top1 is None: val_top1 = trn_top1
        # genotype
        genotype = model.to_genotype()
        genotypes.append(genotype)
        save_genotype(expman, genotype, epoch, logger)
        # genotype as a image
        if config.plot:
            for i, dag in enumerate(model.dags()):
                plot_path = expman.join('plot', "EP{:02d}".format(epoch+1))
                caption = "Epoch {} - DAG {}".format(epoch+1, i)
                plot(genotype.dag[i], dag, plot_path + "-dag_{}".format(i), caption)
        
        if best_top1 < val_top1:
            best_top1 = val_top1
            best_genotype = genotype

        if config.save_freq != 0 and epoch % config.save_freq == 0:
            save_checkpoint(expman, model, w_optim, arch_optim, lr_scheduler, epoch, logger)

        lr_scheduler.step()
        
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    gt.to_file(best_genotype, expman.join('output', 'best.gt'))
    return best_top1, best_genotype, genotypes


def augment(config, chkpt_path, expman, train_loader, valid_loader, model, writer, logger, device):
    w_optim = utils.get_optim(model.weights(), config.w_optim)
    lr_scheduler = utils.get_lr_scheduler(w_optim, config.lr_scheduler, config.epochs)

    init_epoch = -1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        w_optim.load_state_dict(checkpoint['w_optim'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']
    else:
        logger.info("Starting new training run")

    logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))
    
    # training loop
    logger.info('begin training')
    best_top1 = 0.
    tot_epochs = config.epochs
    for epoch in itertools.count(init_epoch+1):
        if epoch == tot_epochs: break

        drop_prob = config.drop_path_prob * epoch / tot_epochs
        model.drop_path_prob(drop_prob)

        lr = lr_scheduler.get_lr()[0]

        # training
        trn_top1 = train(train_loader, None, model, writer, logger, None, w_optim, lr, epoch, tot_epochs, device, config)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        val_top1 = validate(valid_loader, model, writer, logger, epoch, tot_epochs, cur_step, device, config)
        if val_top1 is None: val_top1 = trn_top1

        # save
        if best_top1 < val_top1:
            best_top1 = val_top1
            is_best = True
        else:
            is_best = False
        
        if config.save_freq != 0 and epoch % config.save_freq == 0:
            save_checkpoint(expman, model, w_optim, None, lr_scheduler, epoch, logger)

        lr_scheduler.step()
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1


def train(train_loader, valid_loader, model, writer, logger, arch_optim, w_optim, lr, epoch, tot_epochs, device, config):
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
        # phase 1. child network step
        tprof.timer_start('train')
        w_optim.zero_grad()
        loss, logits = model.loss_logits(trn_X, trn_y, config.aux_weight)
        loss.backward()
        # gradient clipping
        if config.w_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        tprof.timer_stop('train')
        # phase 2. arch_optim step
        if update_arch and (step+1) % arch_update_intv == 0:
            for a_batch in range(arch_update_batch):
                if valid_loader is None:
                    tprof.timer_start('arch')
                    arch_optim.step(trn_X, trn_y, trn_X, trn_y, lr, w_optim)
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
                    arch_optim.step(trn_X, trn_y, val_X, val_y, lr, w_optim)
                tprof.timer_stop('arch')

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step !=0 and step % config.print_freq == 0 or step == n_trn_batch-1:
            eta = eta_m.step(step)
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} LR {:.3f} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%}) | ETA: {eta}".format(
                    epoch+1, tot_epochs, step, n_trn_batch-1, lr, losses=losses,
                    top1=top1, top5=top5, eta=utils.format_time(eta)))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1
        if step < n_trn_batch-1: tprof.timer_start('data')
    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.print_stat('data')
    tprof.print_stat('train')
    tprof.print_stat('arch')
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
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, tot_epochs, step, n_val_batch-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, tot_epochs, top1.avg))
    tprof.print_stat('validate')

    return top1.avg
