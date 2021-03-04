import numpy as np
from modnas.core.param_space import ParamSpace
from modnas.core.params import Categorical
from modnas.registry.estim import RegressionEstim
from modnas.registry.construct import register as register_constructor
from modnas.registry.estim import register as register_estim


@register_constructor
class FakeDataSpaceConstructor():
    def __init__(self, n_nodes=2**10, dim=2**1):
        self.n_nodes = n_nodes
        self.dim = dim

    def __call__(self, model):
        del model
        _ = [Categorical(list(range(self.dim))) for _ in range(self.n_nodes)]


class FakeDataPredictor():
    def __init__(self, score_dim=1, seed=11235, random_score=False, noise_scale=0.01):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.score_dim = score_dim
        self.random_score = random_score
        self.noise_scale = noise_scale
        self.scores = {'dim_{}'.format(i): {} for i in range(score_dim)}

    def fit(self, ):
        pass

    def get_score(self, params, scores):
        score = 0
        for pn, v in params.items():
            p = ParamSpace().get_param(pn)
            idx = p.get_index(v)
            dim = len(p)
            if pn not in scores:
                if self.random_score:
                    p_score = self.rng.rand(dim)
                    p_score = p_score / np.max(p_score)
                else:
                    p_score = list(range(dim))
                scores[pn] = p_score
            score += scores[pn][idx]
        score /= len(params)
        score += 0 if self.noise_scale is None else self.rng.normal(loc=0, scale=self.noise_scale)
        return score

    def predict(self, params):
        scores = {k: self.get_score(params, v) for k, v in self.scores.items()}
        if len(scores) == 1:
            return list(scores.values())[0]
        return scores


@register_estim
class FakeDataEstim(RegressionEstim):
    def __init__(self, *args, pred_conf=None, **kwargs):
        super().__init__(*args, predictor=FakeDataPredictor(**(pred_conf or {})), **kwargs)

    def run(self, optim):
        ret = super().run(optim)
        scores = self.predictor.scores
        if scores and len(scores) == 1:
            scores = np.array(list(list(scores.values())[0].values()))
            self.logger.info('global optimum arch_desc: {}, score: {}'.format(scores.argmax(1),
                                                                              sum(np.max(scores, 1)) / scores.shape[0]))
        return ret
