import time
import numpy as np
from collections import OrderedDict
from modnas.optim import register
from modnas.optim.base import OptimBase
from modnas.core.param_space import ParamCategorical, ParamNumeric
try:
    import skopt
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
except ImportError:
    skopt = None


@register
class SkoptOptim(OptimBase):
    def __init__(self, space, skopt_args={}, logger=None):
        super().__init__(space, logger)
        if skopt is None:
            raise ValueError('scikit-optimize is not installed')
        skopt_dims = []
        param_names = []
        for n, p in self.space.named_params():
            if isinstance(p, ParamNumeric):
                if p.is_int():
                    sd = Integer(*p.bound, name=n)
                else:
                    sd = Real(*p.bound, name=n)
            elif isinstance(p, ParamCategorical):
                sd = Categorical(p.choices, name=n)
            else:
                continue
            skopt_dims.append(sd)
            param_names.append(n)
        skopt_args['dimensions'] = skopt_dims
        if 'random_state' not in skopt_args:
            skopt_args['random_state'] = int(time.time())
        self.param_names = param_names
        self.skoptim = Optimizer(**skopt_args)

    def has_next(self):
        return True

    def convert_param(self, p):
        if isinstance(p, np.float):
            return float(p)
        return p

    def _next(self):
        next_pt = self.skoptim.ask()
        next_params = OrderedDict()
        for n, p in zip(self.param_names, next_pt):
            next_params[n] = self.convert_param(p)
        return next_params

    def next(self, batch_size):
        if batch_size == 1:
            return [self._next()]
        next_pts = self.skoptim.ask(n_points=batch_size)
        next_params = []
        for pt in next_pts:
            params = OrderedDict()
            for n, p in zip(self.param_names, pt):
                params[n] = self.convert_param(p)
            next_params.append(params)
        return next_params

    def step(self, estim):
        def to_metrics(res):
            if isinstance(res, dict):
                return list(res.values())[0]
            if isinstance(res, (tuple, list)):
                return res[0]
            return res

        inputs, results = estim.get_last_results()
        skinputs = [list(inp.values()) for inp in inputs]
        skresults = [-to_metrics(r) for r in results]
        self.skoptim.tell(skinputs, skresults)
