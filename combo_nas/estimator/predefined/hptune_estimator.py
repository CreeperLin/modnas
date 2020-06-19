import copy
import itertools
import traceback
from ..base import EstimatorBase
from ...utils.config import Config

class HPTuneEstimator(EstimatorBase):
    def __init__(self, measure_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_fn = measure_fn
        self.results_all = list()
        self.best_hparams = None
        self.best_score = 0
        self.best_iter = 0
        self.trial_index = 0

    def step(self, hp):
        logger = self.logger
        logger.info('measuring hparam: {}'.format(hp))
        config = self.config
        trial_config = copy.deepcopy(Config.load(config.trial_config))
        Config.apply(trial_config, hp)
        measure_args = copy.deepcopy(config.trial_args)
        measure_args.name = '{}_{}'.format(measure_args.get('name', 'trial'), self.trial_index)
        measure_args.exp = self.expman.subdir(measure_args.get('exp', ''))
        measure_args.config = trial_config
        self.trial_index += 1
        try:
            score = self.measure_fn(**measure_args)
            error_no = 0
        except:
            score = 0
            error_no = 1
            logger.info('trial {} failed with error: {}'.format(self.trial_index, traceback.format_exc()))
        result = {
            'score': score,
            'error_no': error_no,
        }
        return result

    def search(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        batch_size = config.get('batch_size', 1)
        early_stopping = config.get('early_stopping', None)

        logger.info('hptune: start: epochs: {}'.format(tot_epochs))
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            if not optim.has_next():
                logger.info('hptune: optim stop at epoch: {}'.format(epoch))
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
            logger.info('hptune: epoch: {}\t score: {:.4f}/{:.4f}'.format(epoch+1, score, self.best_score))
            optim.step(self)
            if not early_stopping is None and epoch >= self.best_iter + early_stopping:
                logger.info('hptune: early stopped: {}'.format(epoch))
                break
        return {
            'best_iter': self.best_iter,
            'best_score': self.best_score,
            'best_hparams': self.best_hparams,
            'results_all': self.results_all,
        }
