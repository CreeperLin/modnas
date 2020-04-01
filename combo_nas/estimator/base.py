# import logging
import torch
import torch.nn as nn
from .. import utils
from ..metrics import build as build_metrics
from ..utils.criterion import build as build_criterion
from ..utils.profiling import tprof
from ..arch_space.ops import Identity, DropPath_
from ..arch_space.constructor import Slot
from ..arch_space import genotypes as gt

class EstimatorBase():
    def __init__(self, config, expman, train_loader, valid_loader,
                 model_builder, model, writer, logger, device):
        self.config = config
        self.expman = expman
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_builder = model_builder
        self.model = model
        if self.model is None and not model_builder is None:
            try:
                self.model = model_builder()
            except Exception as e:
                logger.info('Model build failed: {}'.format(e))
        self.writer = writer
        self.logger = logger
        self.device = device
        self.init_epoch = -1
        self.w_optim = None
        self.lr_scheduler = None
        metrics = {}
        if 'metrics' in config:
            mt_configs = config.metrics
            for mt_name, mt_conf in mt_configs.items():
                metrics_args = mt_conf.get('args', {})
                metrics[mt_name] = build_metrics(mt_conf.type, self.logger, **metrics_args)
        self.metrics = metrics
        criterions_all = []
        criterions_train = []
        criterions_eval = []
        criterions_valid = []
        if 'criterion' in config:
            crit_configs = config.criterion
            if not isinstance(crit_configs, list):
                crit_configs = [crit_configs]
            for crit_conf in crit_configs:
                if isinstance(crit_conf, str):
                    crit = build_criterion(crit_conf)
                    crit_mode = 'all'
                else:
                    crit = build_criterion(crit_conf.type, **crit_conf.get('args', {}))
                    crit_mode = crit_conf.get('mode', 'all')
                if not isinstance(crit_mode, list):
                    crit_mode = [crit_mode]
                if 'all' in crit_mode:
                    criterions_all.append(crit)
                if 'train' in crit_mode:
                    criterions_train.append(crit)
                if 'eval' in crit_mode:
                    criterions_eval.append(crit)
                if 'valid' in crit_mode:
                    criterions_valid.append(crit)
        self.criterions_all = criterions_all
        self.criterions_train = criterions_train
        self.criterions_eval = criterions_eval
        self.criterions_valid = criterions_valid
        self.results = []
        self.inputs = []

    def criterion(self, X, y_pred, y_true, mode=None):
        loss = None
        crits = []
        if not mode is None:
            if mode == 'train':
                crits = self.criterions_train
            elif mode == 'eval':
                crits = self.criterions_eval
            elif mode == 'valid':
                crits = self.criterions_valid
        crits = self.criterions_all + crits
        for i, crit in enumerate(crits):
            if i == 0:
                loss = crit(y_pred, y_true) # basic criterion
            else:
                loss = crit(loss, self, X, y_pred, y_true)
        return loss

    def loss(self, X, y, model=None, mode=None):
        model = self.model if model is None else model
        return self.criterion(X, model.logits(X), y, mode=mode)

    def loss_logits(self, X, y, model=None, mode=None):
        model = self.model if model is None else model
        logits = model.logits(X)
        return self.criterion(X, logits, y, mode=mode), logits

    def print_model_info(self, model=None):
        model = self.model if model is None else model
        if not model is None:
            self.logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(
                utils.param_count(model), utils.param_size(model)))

    def get_last_results(self):
        return self.inputs, self.results

    def compute_metrics(self, *args, name=None, model=None, **kwargs):
        model = self.model if model is None else model
        if not name is None:
            return self.metrics[name].compute(model, *args, **kwargs)
        ret = {}
        for mt_name, mt in self.metrics.items():
            res = mt.compute(model, *args, **kwargs)
            ret[mt_name] = res
        return ret

    def predict(self, ):
        pass

    def train(self):
        pass

    def validate(self, ):
        pass

    def search(self, optim):
        pass

    def train_epoch(self, epoch, tot_epochs, train_loader=None, model=None,
                    writer=None, logger=None, w_optim=None, lr_scheduler=None,
                    device=None, config=None):
        train_loader = self.train_loader if train_loader is None else train_loader
        model = self.model if model is None else model
        writer = self.writer if writer is None else writer
        logger = self.logger if logger is None else logger
        w_optim = self.w_optim if w_optim is None else w_optim
        lr_scheduler = self.lr_scheduler if lr_scheduler is None else lr_scheduler
        device = self.device if device is None else device
        config = self.config if config is None else config
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
            loss, logits = self.loss_logits(trn_X, trn_y, model=model, mode='train')
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

    def validate_epoch(self, epoch, tot_epochs, cur_step=0, valid_loader=None,
                       model=None, writer=None, logger=None, device=None, config=None):
        valid_loader = self.valid_loader if valid_loader is None else valid_loader
        if valid_loader is None: return None
        model = self.model if model is None else model
        writer = self.writer if writer is None else writer
        logger = self.logger if logger is None else logger
        device = self.device if device is None else device
        config = self.config if config is None else config
        if cur_step is None: cur_step = (epoch+1) * len(self.train_loader)
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
                loss, logits = self.loss_logits(val_X, val_y, model=model, mode='eval')
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

    def update_drop_path_prob(self, epoch, tot_epochs, model=None):
        model = self.model if model is None else model
        drop_prob = self.config.drop_path_prob * epoch / tot_epochs
        model.drop_path_prob(drop_prob)
        self.logger.debug('drop path prob: {:.5f}'.format(drop_prob))

    def apply_drop_path(self, model=None):
        if self.config.drop_path_prob <= 0.0: return
        model = self.model if model is None else model
        def apply(slot):
            ent = slot.ent
            if slot.fixed: return
            if ent is None: return
            if not isinstance(ent, Identity):
                ent = nn.Sequential(
                    ent,
                    DropPath_()
                )
            slot.set_entity(ent)
        Slot.apply_all(apply, gen=(lambda: Slot.slots_model(model)))

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}

    def save(self, epoch):
        self.save_genotype(epoch)
        self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch, save_name=None):
        expman = self.expman
        model = self.model
        w_optim = self.w_optim
        lr_scheduler = self.lr_scheduler
        logger = self.logger
        if save_name is None:
            save_name = 'chkpt_{:03d}.pt'.format(epoch+1)
        else:
            save_name = 'chkpt_{}.pt'.format(save_name)
        save_path = expman.join('chkpt', save_name)
        try:
            torch.save({
                'model': model.net.state_dict(),
                'w_optim': w_optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'estim': self.state_dict(),
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
        except Exception as exc:
            logger.error("Save checkpoint failed: "+str(exc))

    def save_genotype(self, epoch, genotype=None, save_name=None):
        expman = self.expman
        logger = self.logger
        genotype = self.model.to_genotype() if genotype is None else genotype
        if save_name is None:
            save_name = 'gene_{:03d}.gt'.format(epoch+1)
        else:
            save_name = 'gene_{}.gt'.format(save_name)
        save_path = expman.join('output', save_name)
        try:
            logger.info("genotype = {}".format(genotype))
            gt.to_file(genotype, save_path)
            logger.info("Saved genotype to: %s" % save_path)
        except Exception as exc:
            logger.error("Save genotype failed: "+str(exc))

    def load(self, chkpt_path):
        if chkpt_path is None:
            self.logger.info("Estimator: Starting new run")
            return
        self.logger.info("Estimator: Resuming from checkpoint: {}".format(chkpt_path))
        chkpt = torch.load(chkpt_path)
        if 'model' in chkpt and not self.model is None:
            self.model.net.load_state_dict(chkpt['model'])
        if 'w_optim' in chkpt and not self.w_optim is None:
            self.w_optim.load_state_dict(chkpt['w_optim'])
        if 'lr_scheduler' in chkpt and not self.lr_scheduler is None:
            self.lr_scheduler.load_state_dict(chkpt['lr_scheduler'])
        if 'estim' in chkpt:
            self.load_state_dict(chkpt['estim'])
        self.init_epoch = chkpt['epoch']
