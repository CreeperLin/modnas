from ...utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('cost_model')

from .sklearn import SKLearnCostModel
from .xgboost import XGBoostCostModel

register(SKLearnCostModel, 'sklearn')
register(XGBoostCostModel, 'xgboost')
