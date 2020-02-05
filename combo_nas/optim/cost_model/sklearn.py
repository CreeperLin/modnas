import importlib
import numpy as np
try:
    import sklearn
except ImportError:
    sklearn = None
from .base import CostModel

class SKLearnCostModel(CostModel):
    def __init__(self, space, model_cls, module, model_kwargs={}):
        super().__init__(space)
        if sklearn is None:
            raise RuntimeError('sklearn is not installed')
        module = importlib.import_module(module)
        model_cls = getattr(module, model_cls)
        self.model = model_cls(**model_kwargs)

    def fit(self, inputs, results):
        x_train = self.to_feature(inputs)
        y_train = np.array(results)
        index = np.random.permutation(len(x_train))
        trn_x, trn_y = x_train[index], y_train[index]
        self.model.fit(trn_x, trn_y)

    def predict(self, inputs):
        feats = self.to_feature(inputs)
        return self.model.predict(feats)
