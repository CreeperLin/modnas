"""Base Estimator."""
import traceback
import pickle
from .. import utils
from ..utils.criterion import build_criterions_all
from ..metrics import build_metrics_all
from ..arch_space.export import build as build_exporter


class EstimBase():
    """Base Estimator class."""

    def __init__(self,
                 config=None,
                 expman=None,
                 trainer=None,
                 constructor=None,
                 exporter=None,
                 model=None,
                 writer=None,
                 logger=None,
                 name=None,
                 profiling=False):
        self.name = '' if name is None else name
        self.config = config
        self.expman = expman
        self.constructor = constructor
        self.exporter = exporter
        self.model = model
        self.writer = writer
        self.logger = logger
        self.cur_epoch = -1
        self.metrics = build_metrics_all(config.get('metrics', None), self, logger)
        self.criterions_all, self.criterions_train, self.criterions_eval, self.criterions_valid = build_criterions_all(
            config.get('criterion', None), getattr(model, 'device_ids', None))
        self.trainer = trainer
        self.results = []
        self.inputs = []
        self.cur_trn_batch = None
        self.cur_val_batch = None

    def set_trainer(self, trainer):
        """Set current trainer."""
        self.trainer = trainer

    def criterion(self, X, y_pred, y_true, model=None, mode=None):
        """Return loss."""
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
        if hasattr(self.trainer, 'loss'):
            loss = self.trainer.loss(model(X) if y_pred is None else y_pred, y_true)
        for crit in crits:
            loss = crit(loss, self, model, X, y_pred, y_true)
        return loss

    def loss(self, X, y, output=None, model=None, mode=None):
        """Return loss."""
        model = self.model if model is None else model
        return self.criterion(X, output, y, model, mode)

    def loss_logits(self, X, y, model=None, mode=None):
        """Return loss and logits."""
        model = self.model if model is None else model
        output = model(X)
        return self.loss(X, y, output, model, mode), output

    def step(self, params):
        """Return evaluation results of a parameter set."""
        raise NotImplementedError

    def print_model_info(self):
        """Output model information."""
        model = self.model
        if model is not None:
            self.logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model, factor=2),
                                                                                    utils.param_size(model, factor=2)))

    def get_last_results(self):
        """Return last evaluation results."""
        return self.inputs, self.results

    def compute_metrics(self, *args, name=None, model=None, to_scalar=True, **kwargs):
        """Return Metrics results."""
        def fmt_key(n, k):
            return '{}.{}'.format(n, k)

        def flatten_dict(n, r):
            if isinstance(r, dict):
                return {fmt_key(n, k): flatten_dict(fmt_key(n, k), v) for k, v in r.items()}
            return r

        def merge_results(dct, n, r):
            if not isinstance(r, dict):
                r = {n: r}
            r = {k: None if v is None else (float(v) if to_scalar else v) for k, v in r.items()}
            dct.update(r)

        ret = {}
        model = self.model if model is None else model
        names = [name] if name is not None else self.metrics.keys()
        for mt_name in names:
            res = self.metrics[mt_name].compute(model, *args, **kwargs)
            merge_results(ret, mt_name, flatten_dict(mt_name, res))
        return ret

    def run(self, optim):
        """Run Estimator routine."""
        raise NotImplementedError

    def get_score(self, res):
        """Return scalar value from evaluation results."""
        if not isinstance(res, dict):
            return res
        score = res.get('default', None)
        if score is None:
            score = 0 if len(res) == 0 else list(res.values())[0]
        return score

    def train_epoch(self, epoch, tot_epochs, model=None):
        """Train model for one epoch."""
        model = self.model if model is None else model
        ret = self.trainer.train_epoch(estim=self,
                                       model=model,
                                       tot_steps=self.get_num_train_batch(epoch),
                                       epoch=epoch,
                                       tot_epochs=tot_epochs)
        return ret

    def train_step(self, epoch, tot_epochs, step, tot_steps, model=None):
        """Train model for one step."""
        model = self.model if model is None else model
        return self.trainer.train_step(estim=self,
                                       model=model,
                                       epoch=epoch,
                                       tot_epochs=tot_epochs,
                                       step=step,
                                       tot_steps=tot_steps)

    def valid_epoch(self, epoch=0, tot_epochs=1, model=None):
        """Validate model for one epoch."""
        model = self.model if model is None else model
        return self.trainer.valid_epoch(estim=self,
                                        model=model,
                                        tot_steps=self.get_num_valid_batch(epoch),
                                        epoch=epoch,
                                        tot_epochs=tot_epochs)

    def valid_step(self, epoch, tot_epochs, step, tot_steps, model=None):
        """Validate model for one step."""
        model = self.model if model is None else model
        return self.trainer.valid_step(estim=self,
                                       model=model,
                                       epoch=epoch,
                                       tot_epochs=tot_epochs,
                                       step=step,
                                       tot_steps=tot_steps)

    def reset_trainer(self,
                      tot_epochs=None,
                      config=None,
                      device=None,
                      optimizer_config=None,
                      lr_scheduler_config=None,
                      model=None,
                      scale_lr=True):
        """Reinitialize trainer."""
        model = self.model if model is None else model
        config = self.config if config is None else config
        tot_epochs = config.epochs if tot_epochs is None else tot_epochs
        if self.trainer is not None:
            self.trainer.init(model,
                              optimizer_config=optimizer_config,
                              tot_epochs=tot_epochs,
                              scale_lr=scale_lr,
                              lr_scheduler_config=lr_scheduler_config,
                              device=device)
        self.cur_epoch = -1

    def get_num_train_batch(self, epoch=None):
        """Return number of training batches."""
        epoch = self.cur_epoch if epoch is None else epoch
        return 0 if self.trainer is None else self.trainer.get_num_train_batch(epoch=epoch)

    def get_num_valid_batch(self, epoch=None):
        """Return number of validating batches."""
        epoch = self.cur_epoch if epoch is None else epoch
        return 0 if self.trainer is None else self.trainer.get_num_valid_batch(epoch=epoch)

    def get_next_train_batch(self):
        """Return the next training batch."""
        ret = self.trainer.get_next_train_batch()
        self.cur_trn_batch = ret
        return ret

    def get_cur_train_batch(self):
        """Return the current training batch."""
        return self.cur_trn_batch or self.get_next_train_batch()

    def get_next_valid_batch(self):
        """Return the next validating batch."""
        ret = self.trainer.get_next_valid_batch()
        self.cur_val_batch = ret
        return ret

    def get_cur_valid_batch(self):
        """Return the current validating batch."""
        return self.cur_val_batch

    def load_state_dict(self, state_dict):
        """Resume states."""
        pass

    def state_dict(self):
        """Return current states."""
        return {'cur_epoch': self.cur_epoch}

    def get_arch_desc(self):
        """Return current archdesc."""
        return self.exporter(self.model)

    def save_model(self, save_name=None, exporter='DefaultTorchCheckpointExporter'):
        """Save model checkpoint to file."""
        expman = self.expman
        save_name = 'model_{}_{}.pt'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        build_exporter(exporter, path=chkpt_path)(self.model)

    def save(self, epoch=None, save_name=None):
        """Save Estimator states to file."""
        expman = self.expman
        logger = self.logger
        save_name = 'estim_{}_{}.pkl'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        epoch = epoch or self.cur_epoch
        try:
            chkpt = self.state_dict()
            with open(chkpt_path, 'wb') as f:
                pickle.dump(chkpt, f)
        except RuntimeError:
            logger.error("Failed saving estimator: {}".format(traceback.format_exc()))

    def save_checkpoint(self, epoch=None, save_name=None):
        """Save Estimator & model to file."""
        epoch = epoch or self.cur_epoch
        save_name = save_name or 'ep{:03d}'.format(epoch + 1)
        self.save_model(save_name)
        self.save(epoch, save_name)

    def save_arch_desc(self, epoch=None, arch_desc=None, save_name=None, exporter='DefaultToFileExporter'):
        """Save archdesc to file."""
        expman = self.expman
        logger = self.logger
        if save_name is not None:
            fname = 'arch_{}_{}'.format(self.name, save_name)
        else:
            epoch = epoch or self.cur_epoch
            fname = 'arch_{}_ep{:03d}'.format(self.name, epoch + 1)
        save_path = expman.join('output', fname)
        try:
            build_exporter(exporter, path=save_path)(arch_desc)
        except RuntimeError:
            logger.error("Failed saving arch_desc: {}".format(traceback.format_exc()))

    def load(self, chkpt_path):
        """Load states from file."""
        if chkpt_path is None:
            return
        self.logger.info("Resuming from checkpoint: {}".format(chkpt_path))
        with open(chkpt_path, 'rb') as f:
            chkpt = pickle.load(f)
        self.load_state_dict(chkpt)
