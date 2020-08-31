import random
from combo_nas.estim.predefined.default import DefaultEstimator
from combo_nas.estim import register_as
from combo_nas.core.param_space import ArchParamSpace


@register_as('RandomSampling')
class RandomSamplingEstimator(DefaultEstimator):
    def __init__(self, *args, seed=1, save_best=True, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        random.seed(seed)
        self.space_size = ArchParamSpace.categorical_size()

    def loss(self, X, y, output=None, model=None, mode=None):
        model = self.model if model is None else model
        loss = self.criterion(X, output, y, model, mode)
        params = ArchParamSpace.get_categorical_params(random.randint(0, self.space_size - 1))
        ArchParamSpace.update_params(params)
        return loss
