# -*- coding: utf-8 -*-
from .gridsearch_tuner import GridSearchTuner, RandomTuner
from .xgb_tuner import XGBoostTuner
from .space import build_hparam_space
from ..utils.registration import Registry, build, get_builder, register, register_wrapper
from functools import partial

hparam_tuner_registry = Registry('hparam_tuner')
register_hparam_tuner = partial(register, hparam_tuner_registry)
get_hparam_tuner_builder = partial(get_builder, hparam_tuner_registry)
build_hparam_tuner = partial(build, hparam_tuner_registry)
register = partial(register_wrapper, hparam_tuner_registry)

register_hparam_tuner(GridSearchTuner, 'GridSearch')
register_hparam_tuner(RandomTuner, 'Random')
register_hparam_tuner(XGBoostTuner, 'XGBoost')