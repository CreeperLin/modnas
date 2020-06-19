import random
from combo_nas.estimator.predefined.default_estimator import DefaultEstimator
from combo_nas.estimator import register_as
from combo_nas.core.param_space import ArchParamSpace

@register_as('RandomSampling')
class RandomSamplingEstimator(DefaultEstimator):
    def __init__(self, *args, seed=1, save_best=False, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        random.seed(seed)
        self.space_size = ArchParamSpace.categorical_size()

    def loss_logits(self, X, y, model=None, mode=None):
        model = self.model if model is None else model
        params = ArchParamSpace.get_categorical_params(random.randint(0, self.space_size - 1))
        ArchParamSpace.update_params(params)
        logits = model.logits(X)
        return self.criterion(X, logits, y, model=model, mode=mode), logits
