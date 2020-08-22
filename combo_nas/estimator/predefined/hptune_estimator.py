import copy
import itertools
import traceback
from ..base import EstimatorBase
from ... import utils
from ...utils.config import Config
from ...utils.wrapper import build as build_runner

class HPTuneEstimator(EstimatorBase):
    def __init__(self, measure_fn=None, batch_size=1, early_stopping=None,
                 trial_proc=None, trial_config=None, trial_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_fn = measure_fn or self.default_measure_fn
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.trial_proc = trial_proc
        self.trial_config = trial_config
        self.trial_args = trial_args
        self.results_all = list()
        self.best_hparams = None
        self.best_score = 0
        self.best_iter = 0
        self.trial_index = 0

    def default_measure_fn(self, hp, **kwargs):
        trial_config = copy.deepcopy(Config.load(self.trial_config))
        Config.apply(trial_config, hp)
        trial_args = copy.deepcopy(self.trial_args)
        trial_args.name = '{}_{}'.format(trial_args.get('name', 'trial'), self.trial_index)
        trial_args.exp = self.expman.subdir(trial_args.get('exp', ''))
        trial_args.config = trial_config
        ret = build_runner(self.trial_proc, **trial_args)
        return ret['final'].get('best_score', list(ret.values())[0])

    def step(self, hp):
        self.trial_index += 1
        logger = self.logger
        logger.info('measuring hparam: {}'.format(hp))
        config = self.config
        fn_args = config.get('trial_args', {})
        try:
            score = self.measure_fn(hp, **fn_args)
            error_no = 0
        except:
            score = 0
            error_no = 1
            logger.info('trial {} failed with error: {}'.format(self.trial_index, traceback.format_exc()))
        result = {
            'score': score or 0,
            'error_no': error_no,
        }
        return result

    def run(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        batch_size = self.batch_size
        early_stopping = self.early_stopping
        if tot_epochs > 0:
            eta_m = utils.ETAMeter(tot_epochs, self.cur_epoch)
            eta_m.start()
        else:
            eta_m = None
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            if not optim.has_next():
                logger.info('HPTune: all finished')
                break
            self.inputs = optim.next(batch_size)
            self.results = []
            for hp in self.inputs:
                res = self.step(hp)
                score = 0 if res['error_no'] else res['score']
                if score > self.best_score:
                    self.best_score = score
                    self.best_hparams = hp
                    self.best_iter = epoch
                self.results.append(score)
                self.results_all.append((hp, score))
            if not eta_m is None:
                eta_m.step()
                eta = eta_m.eta_fmt()
            else:
                eta = 'N/A'
            logger.info('HPTune: [{:3d}/{}] Current: {:.4f} Best: {:.4f} | ETA: {}'.format(
                epoch+1, tot_epochs, score, self.best_score or 0, eta))
            optim.step(self)
            if not early_stopping is None and epoch >= self.best_iter + early_stopping:
                logger.info('HPTune: early stopped: {}'.format(epoch))
                break
        return {
            'best_iter': self.best_iter,
            'best_score': self.best_score,
            'best_hparams': self.best_hparams,
            'results_all': self.results_all,
        }
