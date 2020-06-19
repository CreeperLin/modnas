import numpy as np
import torch.nn as nn
from combo_nas.core.param_space import ArchParamSpace, ArchParamCategorical
from combo_nas.estimator.predefined.regression_estimator import RegressionEstimator, ArchPredictor
import combo_nas.arch_space
import combo_nas.estimator

@combo_nas.arch_space.register_as('FakeData')
class FakeDataNet(nn.Module):
    def __init__(self, n_nodes=2**10, dim=2**1):
        super().__init__()
        matrix = []
        for _ in range(n_nodes):
            matrix.append(ArchParamCategorical(list(range(dim))))
        self.matrix_params = matrix


class FakeDataPredictor(ArchPredictor):
    def __init__(self, seed=11235, random_score=False, noise_scale=0.01):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.random_score = random_score
        self.noise_scale = noise_scale
        self.scores = {}

    def fit(self, ):
        pass

    def predict(self, params):
        score = 0
        for pn, v in params.items():
            p = ArchParamSpace.get_param(pn)
            idx = p.get_index(v)
            dim = len(p)
            if not pn in self.scores:
                if self.random_score:
                    p_score = self.rng.rand(dim)
                    p_score = p_score / np.max(p_score)
                else:
                    p_score = list(range(dim))
                # p_score = np.exp(p_score) / np.sum(np.exp(p_score))
                self.scores[pn] = p_score
            score += self.scores[pn][idx]
        score /= len(params)
        score += 0 if self.noise_scale is None else self.rng.normal(loc=0, scale=self.noise_scale)
        return score


@combo_nas.estimator.register_as('FakeData')
class FakeDataEstimator(RegressionEstimator):

    def search(self, optim):
        config = self.config
        self.predictor = FakeDataPredictor()
        self.model = None
        ret = super().search(optim)
        scores = np.array(list(self.predictor.scores.values()))
        self.logger.info('global optimum genotype: {}, score: {}'.format(
                         scores.argmax(1), sum(np.max(scores, 1)) / scores.shape[0]))
        return ret
