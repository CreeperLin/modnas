import logging
import json
from ..core.param_space import ParamSpace, Param, ParamDiscrete, ParamContinuous

class HParamSpaceClass(ParamSpace):
    def to_json(self, path):
        json.dump(dict(self._params_map), path)


HParamSpace = HParamSpaceClass()


class HParamDiscrete(ParamDiscrete, Param):
    def __init__(self, valrange, sampler=None, name=None):
        ParamDiscrete.__init__(self, valrange, sampler)
        Param.__init__(self, HParamSpace, name)


class HParamContinuous(ParamContinuous, Param):
    def __init__(self, shape, sampler=None, name=None):
        ParamContinuous.__init__(self, shape, sampler)
        Param.__init__(self, HParamSpace, name)


def build_hparam_space_from_json(path):
    hp_json = json.load(open(path, 'r'))
    build_hparam_space_from_dict(dict(hp_json))


def build_hparam_space_from_dict(hp_dict):
    for k, v in hp_dict.items():
        if isinstance(v, list):
            hp = HParamDiscrete(name=k, valrange=v)
        else:
            raise ValueError('support hparam in list format only')
    logging.debug('hparam: space size: {}'.format(HParamSpace.discrete_size()))