from . import register
from ...core.param_space import ParamNumeric, ParamCategorical


@register
class DefaultHParamSpaceConstructor():
    def __init__(self, params):
        if isinstance(params, dict):
            params = params.items()
        elif isinstance(params, list):
            params = [(None, p) for p in params]
        self.params = params

    def __call__(self, model):
        del model
        for k, v in self.params:
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                _ = ParamNumeric(low=v[0][0], high=v[0][1], name=k)
            elif isinstance(v, list):
                _ = ParamCategorical(choices=v, name=k)
            else:
                raise ValueError('support categorical and numeric hparams only')
