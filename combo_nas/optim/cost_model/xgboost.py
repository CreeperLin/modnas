import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from .base import CostModel

xgb_params_reg = {
    'max_depth': 3,
    'gamma': 0.0001,
    'min_child_weight': 1,
    'subsample': 1.0,
    'eta': 0.3,
    'lambda': 1.00,
    'alpha': 0,
    'objective': 'reg:squarederror',
}

xgb_params_rank = {
    'max_depth': 3,
    'gamma': 0.0001,
    'min_child_weight': 1,
    'subsample': 1.0,
    'eta': 0.3,
    'lambda': 1.00,
    'alpha': 0,
    'objective': 'rank:pairwise',
}

class XGBoostCostModel(CostModel):
    def __init__(self, space, loss_type='reg', xgb_kwargs={}):
        super().__init__(space)
        if xgb is None:
            raise RuntimeError('xgboost is not installed')
        xgb_params = xgb_params_rank if loss_type == 'rank' else xgb_params_reg
        xgb_params.update(xgb_kwargs)
        self.xgb_params = xgb_params
        self.model = None

    def fit(self, inputs, results):
        x_train = self.to_feature(inputs)
        y_train = self.to_target(results)
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self.model = xgb.train(self.xgb_params, dtrain, num_boost_round=400,)

    def predict(self, inputs):
        feats = self.to_feature(inputs)
        dtest = xgb.DMatrix(feats)
        return self.model.predict(dtest)
