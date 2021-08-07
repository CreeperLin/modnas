"""Scikit-learn score prediction model."""
import importlib
import numpy as np
try:
    import sklearn
except ImportError:
    sklearn = None
from .base import ScoreModel
from modnas.registry.score_model import register
from collections import OrderedDict
from numpy import ndarray
from typing import List


@register
class SKLearnScoreModel(ScoreModel):
    """Scikit-learn score prediction model class."""

    def __init__(self, space, model_cls, module, model_kwargs={}):
        super().__init__(space)
        if sklearn is None:
            raise RuntimeError('scikit-learn is not installed')
        module = importlib.import_module(module)
        model_cls = getattr(module, model_cls)
        self.model = model_cls(**model_kwargs)

    def fit(self, inputs: List[OrderedDict], results: List[float]) -> None:
        """Fit model with evaluation results."""
        x_train = self.to_feature(inputs)
        y_train = self.to_target(results)
        index = np.random.permutation(len(x_train))
        trn_x, trn_y = x_train[index], y_train[index]
        self.model.fit(trn_x, trn_y)

    def predict(self, inputs: List[OrderedDict]) -> ndarray:
        """Return predicted evaluation score from model."""
        feats = self.to_feature(inputs)
        return self.model.predict(feats)
