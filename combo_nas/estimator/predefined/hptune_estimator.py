import os
import copy
import torch
import itertools
import traceback
from ..base import EstimatorBase
from ...utils.config import Config

class HPTuneEstimator(EstimatorBase):
    def __init__(self, config, expman, writer, logger, device, measure_fn):
        super().__init__(config, expman, None, None, None, writer, logger, device)
        self.trial_index = 0
        self.measure_fn = measure_fn

    def step(self, hp):
        logger = self.logger
        config = self.config
        trial_config = copy.deepcopy(config.trial_config)
        Config.apply(trial_config, hp)
        measure_args = copy.deepcopy(config.trial_args)
        measure_args.name = '{}_{}'.format(measure_args.get('name', 'trial'), self.trial_index)
        measure_args.exp_root_dir = os.path.join(self.expman.root_dir, measure_args.get('exp_root_dir', ''))
        measure_args.chkpt = measure_args.get('chkpt', None)
        measure_args.device = measure_args.get('device', 'all')
        measure_args.genotype = measure_args.get('genotype', None)
        measure_args.config = trial_config
        self.trial_index += 1
        try:
            score = self.measure_fn(**measure_args)
            error_no = 0
        except Exception as e:
            traceback.print_exc()
            score = 0
            error_no = 1
            logger.debug('trial {} failed with exit code: {}'.format(self.trial_index, error_no))
        result = {
            'score': score,
            'error_no': error_no,
        }
        return result

    def predict(self, ):
        pass
    
    def train(self):
        pass

    def validate(self):
        pass
    
    def get_last_results(self):
        return (self.hparams, self.results)

    def search(self, optim):
        logger = self.logger
        config = self.config
        tr_config = config.trial_config
        batch_size = config.get('batch_size', 1)
        early_stopping = config.get('early_stopping', 1e9)
        epochs = config.epochs
        
        best_hparams = None
        best_score = 0
        best_iter = 0
        ttl = None
        i = error_ct = 0
        logger.info('hptune: start: epochs={} early_stopping={}'.format(epochs, early_stopping))
        for i in range(epochs):
            if not optim.has_next():
                logger.info('hptune: optim stop iter: {}'.format(i))
                break
            self.hparams = optim.next(batch_size)
            self.results = []
            for hp in self.hparams:
                res = self.step(hp)
                # keep best config
                if res['error_no'] == 0:
                    score = res['score']
                    error_ct = 0
                else:
                    score = 0
                    error_ct += 1
                if score > best_score:
                    best_score = score
                    best_hparams = hp
                    best_iter = i
                self.results.append(res)
            logger.info('hptune: iter: {}\t score: {:.4f}/{:.4f}'.format(i+1, score, best_score))
            ttl = min(early_stopping + best_iter, epochs) - i
            optim.update(self)
            if i >= best_iter + early_stopping:
                logger.info('hptune: early stopped: {}'.format(i))
                break
            if error_ct > 150:
                logger.warning('hptune: Too many errors in tuning: {}'.format(error_ct))
        return best_iter, best_score, best_hparams