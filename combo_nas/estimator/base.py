import traceback
import pickle
from .. import utils
from ..metrics import build as build_metrics
from ..metrics.base import MetricsBase
from ..utils.criterion import get_criterion
from ..utils.profiling import TimeProfiler
from ..trainer import build as build_trainer
from ..arch_space import genotypes as gt

class EstimatorBase():
    def __init__(self, config=None, expman=None, data_provider=None,
                 model_builder=None, model=None, writer=None, logger=None,
                 name=None, profiling=False):
        self.name = '' if name is None else name
        self.config = config
        self.expman = expman
        self.data_provider = data_provider
        self.model_builder = model_builder
        self.model = model
        if self.model is None and not model_builder is None:
            try:
                self.model = model_builder()
            except RuntimeError as e:
                logger.info('Model build failed: {}'.format(e))
        self.writer = writer
        self.logger = logger
        self.cur_epoch = -1
        metrics = {}
        mt_configs = config.get('metrics', None)
        if mt_configs:
            MetricsBase.set_estim(self)
            if not isinstance(mt_configs, dict):
                mt_configs = {'default': mt_configs}
            for mt_name, mt_conf in mt_configs.items():
                if isinstance(mt_conf, str):
                    mt_conf = {'type': mt_conf}
                mt_type = mt_conf['type']
                mt_args = mt_conf.get('args', {})
                mt = build_metrics(mt_type, self.logger, **mt_args)
                metrics[mt_name] = mt
        self.metrics = metrics
        criterions_all = []
        criterions_train = []
        criterions_eval = []
        criterions_valid = []
        crit_configs = config.get('criterion', None)
        if crit_configs:
            if not isinstance(crit_configs, list):
                crit_configs = [crit_configs]
            for crit_conf in crit_configs:
                if isinstance(crit_conf, str):
                    crit_conf = {'type': crit_conf}
                try:
                    device_ids = model.device_ids
                except AttributeError:
                    device_ids = None
                crit = get_criterion(crit_conf, device_ids=device_ids)
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
        trainer = None
        trn_conf = config.get('trainer', None)
        if trn_conf:
            if isinstance(trn_conf, str):
                trn_conf = {'type': trn_conf}
            trn_args = {
                'w_optim': config.get('w_optim', None),
                'lr_scheduler': config.get('lr_scheduler', None),
            }
            trn_args.update(trn_conf.get('args', {}))
            trainer = build_trainer(trn_conf['type'], writer=writer, logger=logger, **trn_args)
        self.trainer = trainer
        self.results = []
        self.inputs = []
        self.tprof = TimeProfiler(enabled=profiling)
        self.n_trn_batch = 0 if self.data_provider is None else self.data_provider.get_num_train_batch()
        self.n_val_batch = 0 if self.data_provider is None else self.data_provider.get_num_valid_batch()
        self.cur_trn_batch = None
        self.cur_val_batch = None
        self.no_valid_warn = True

    def criterion(self, X, y_pred, y_true, model=None, mode=None):
        model = self.model if model is None else model
        if mode is None:
            crits = []
        elif mode == 'train':
            crits = self.criterions_train
        elif mode == 'eval':
            crits = self.criterions_eval
        elif mode == 'valid':
            crits = self.criterions_valid
        else:
            raise ValueError('invalid criterion mode: {}'.format(mode))
        crits = self.criterions_all + crits
        loss = None
        for crit in crits:
            loss = crit(loss, self, model, X, y_pred, y_true)
        return loss

    def loss(self, X, y, output=None, model=None, mode=None):
        model = self.model if model is None else model
        return self.criterion(X, output, y, model, mode)

    def loss_logits(self, X, y, model=None, mode=None):
        model = self.model if model is None else model
        logits = model.logits(X)
        return self.criterion(X, logits, y, model, mode), logits

    def print_model_info(self):
        model = self.model
        if not model is None:
            self.logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(
                utils.param_count(model), utils.param_size(model)))

    def get_last_results(self):
        return self.inputs, self.results

    def compute_metrics(self, *args, name=None, model=None, to_scalar=True, **kwargs):
        model = self.model if model is None else model
        if not name is None:
            res = self.metrics[name].compute(model, *args, **kwargs)
            return None if res is None else (float(res) if to_scalar else res)
        ret = {}
        for mt_name, mt in self.metrics.items():
            res = mt.compute(model, *args, **kwargs)
            ret[mt_name] = None if res is None else (float(res) if to_scalar else res)
        return ret

    def predict(self, ):
        pass

    def train(self):
        pass

    def validate(self):
        return self.validate_epoch(epoch=0, tot_epochs=1)

    def search(self, optim):
        pass

    def get_score(self, res):
        if not isinstance(res, dict): return res
        score = res.get('default', None)
        if score is None:
            score = 0 if len(res) == 0 else list(res.values())[0]
        return score

    def get_lr(self):
        return self.trainer.get_lr()

    def get_w_optim(self):
        return self.trainer.w_optim

    def train_epoch(self, epoch, tot_epochs, model=None):
        self.data_provider.reset_train_iter()
        model = self.model if model is None else model
        tprof = self.tprof
        ret = self.trainer.train_epoch(estim=self, model=model, tot_steps=self.n_trn_batch,
                                       epoch=epoch, tot_epochs=tot_epochs)
        tprof.stat('data')
        tprof.stat('train')
        return ret

    def train_step(self, epoch, tot_epochs, step, tot_steps, model=None):
        model = self.model if model is None else model
        return self.trainer.train_step(estim=self, model=model,
                                       epoch=epoch, tot_epochs=tot_epochs,
                                       step=step, tot_steps=tot_steps)

    def validate_epoch(self, epoch=0, tot_epochs=1, model=None):
        self.data_provider.reset_valid_iter()
        model = self.model if model is None else model
        return self.trainer.validate_epoch(estim=self, model=model, tot_steps=self.n_val_batch,
                                           epoch=epoch, tot_epochs=tot_epochs)

    def validate_step(self, epoch, tot_epochs, step, tot_steps, model=None, ):
        model = self.model if model is None else model
        return self.trainer.validate_step(estim=self, model=model,
                                          epoch=epoch, tot_epochs=tot_epochs,
                                          step=step, tot_steps=tot_steps)

    def reset_training_states(self, tot_epochs=None, config=None, device=None,
                              w_optim_config=None, lr_scheduler_config=None,
                              model=None, scale_lr=True):
        model = self.model if model is None else model
        config = self.config if config is None else config
        tot_epochs = config.epochs if tot_epochs is None else tot_epochs
        if not self.trainer is None:
            self.trainer.init(model, w_optim_config=w_optim_config,
                              tot_epochs=tot_epochs, scale_lr=scale_lr,
                              lr_scheduler_config=lr_scheduler_config,
                              device=device)
        self.cur_epoch = -1

    def get_next_trn_batch(self):
        self.tprof.timer_start('data')
        ret = self.data_provider.get_next_train_batch()
        self.tprof.timer_stop('data')
        self.cur_trn_batch = ret
        return ret

    def get_cur_trn_batch(self):
        return self.cur_trn_batch

    def get_next_val_batch(self):
        self.tprof.timer_start('data')
        ret = self.data_provider.get_next_valid_batch()
        self.tprof.timer_stop('data')
        self.cur_val_batch = ret
        return ret

    def get_cur_val_batch(self):
        return self.cur_val_batch

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {
            'cur_epoch': self.cur_epoch
        }

    def save_model(self, save_name=None):
        expman = self.expman
        save_name = 'model_{}_{}.pt'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        self.model.save(chkpt_path)

    def save(self, epoch=None, save_name=None):
        expman = self.expman
        logger = self.logger
        save_name = 'estim_{}_{}.pt'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        epoch = epoch or self.cur_epoch
        try:
            chkpt = self.state_dict()
            with open(chkpt_path, 'wb') as f:
                pickle.dump(chkpt, f)
        except RuntimeError:
            logger.error("Failed saving estimator: {}".format(traceback.format_exc()))

    def save_checkpoint(self, epoch=None, save_name=None):
        epoch = epoch or self.cur_epoch
        save_name = save_name or 'ep{:03d}'.format(epoch+1)
        self.save_model(save_name)
        self.save(epoch, save_name)

    def save_genotype(self, epoch=None, genotype=None, save_name=None):
        expman = self.expman
        logger = self.logger
        genotype = self.model.to_genotype() if genotype is None else genotype
        if not save_name is None:
            fname = 'gene_{}_{}.gt'.format(self.name, save_name)
        else:
            epoch = epoch or self.cur_epoch
            fname = 'gene_{}_ep{:03d}.gt'.format(self.name, epoch+1)
        save_path = expman.join('output', fname)
        try:
            gt.to_file(genotype, save_path)
        except RuntimeError:
            logger.error("Failed saving genotype: {}".format(traceback.format_exc()))

    def load(self, chkpt_path):
        if chkpt_path is None:
            return
        self.logger.info("Resuming from checkpoint: {}".format(chkpt_path))
        with open(chkpt_path, 'rb') as f:
            chkpt = pickle.load(f)
        if 'model' in chkpt and not self.model is None:
            self.model.load(chkpt['model']) # legacy
        if 'states' in chkpt:
            self.load_state_dict(chkpt['states'])
