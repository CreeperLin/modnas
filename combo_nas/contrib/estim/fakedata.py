import numpy as np
from combo_nas.core.param_space import ArchParamSpace, ArchParamCategorical
from combo_nas.estim.predefined.regression import RegressionEstimator, ArchPredictor
from combo_nas.arch_space.construct import register as register_constructor
import combo_nas.estim


@register_constructor
class FakeDataSpaceConstructor():
    def __init__(self, n_nodes=2**10, dim=2**1):
        self.n_nodes = n_nodes
        self.dim = dim

    def __call__(self, model):
        del model
        _ = [ArchParamCategorical(list(range(self.dim))) for _ in range(self.n_nodes)]


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
            if pn not in self.scores:
                if self.random_score:
                    p_score = self.rng.rand(dim)
                    p_score = p_score / np.max(p_score)
                else:
                    p_score = list(range(dim))
                self.scores[pn] = p_score
            score += self.scores[pn][idx]
        score /= len(params)
        score += 0 if self.noise_scale is None else self.rng.normal(loc=0, scale=self.noise_scale)
        return score


@combo_nas.estim.register_as('FakeData')
class FakeDataEstimator(RegressionEstimator):
    def run(self, optim):
        self.predictor = FakeDataPredictor()
        self.model = None
        ret = super().run(optim)
        scores = np.array(list(self.predictor.scores.values()))
        self.logger.info('global optimum arch_desc: {}, score: {}'.format(scores.argmax(1),
                                                                          sum(np.max(scores, 1)) / scores.shape[0]))
        return ret
