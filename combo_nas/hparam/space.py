import logging
from ..core.param_space import ParamSpace, ParamCategorical, ParamNumeric

HParamSpace = ParamSpace()

class HParamCategorical(ParamCategorical):
    def __init__(self, choices, sampler=None, name=None):
        ParamCategorical.__init__(self, HParamSpace, choices, sampler, name)


class HParamNumeric(ParamNumeric):
    def __init__(self, low, high, ntype=None, sampler=None, name=None):
        ParamNumeric.__init__(self, HParamSpace, low, high, ntype, sampler, name)


def build_hparam_space_from_dict(hp_dict):
    for k, v in hp_dict.items():
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            hp = HParamNumeric(*v[0], name=k)
        elif isinstance(v, list):
            hp = HParamCategorical(choices=v, name=k)
        else:
            raise ValueError('support categorical and numeric hparams only')
    logging.debug('hparam: space size: {}'.format(HParamSpace.categorical_size()))
