import copy
import itertools
import traceback
from ..base import EstimatorBase
from ...utils.config import Config

class HPTuneEstimator(EstimatorBase):
    def __init__(self, measure_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_index = 0
        self.measure_fn = measure_fn

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
        except Exception as exc:
            traceback.print_exc()
            score = 0
            error_no = 1
            logger.debug('trial {} failed with error: {}'.format(self.trial_index, exc))
        result = {
            'score': score,
            'error_no': error_no,
        }
        return result

    def load(self, chkpt_path):
        pass

    def predict(self, ):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def search(self, optim):
        logger = self.logger
        config = self.config
        tot_epochs = config.epochs
        batch_size = config.get('batch_size', 1)
        early_stopping = config.get('early_stopping', 1e9)

        best_hparams = None
        best_score = 0
        best_iter = 0
        error_ct = 0
        logger.info('hptune: start: epochs={} early_stopping={}'.format(tot_epochs, early_stopping))
        for epoch in itertools.count(self.cur_epoch+1):
            if epoch == tot_epochs: break
            if not optim.has_next():
                logger.info('hptune: optim stop iter: {}'.format(epoch))
                break
            self.inputs = optim.next(batch_size)
            self.results = []
            for hp in self.inputs:
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
                    best_iter = epoch
                self.results.append(score)
            logger.info('hptune: iter: {}\t score: {:.4f}/{:.4f}'.format(epoch+1, score, best_score))
            optim.step(self)
            if epoch >= best_iter + early_stopping:
                logger.info('hptune: early stopped: {}'.format(epoch))
                break
            if error_ct > 150:
                logger.warning('hptune: Too many errors in tuning: {}'.format(error_ct))
        return {
            'best_iter': best_iter,
            'best_score': best_score,
            'best_hparams': best_hparams
        }
