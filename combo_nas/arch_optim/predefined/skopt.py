import time
from collections import OrderedDict
from ..base import ArchOptimBase
from ...core.param_space import ParamCategorical, ParamNumeric
try:
    import skopt
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
except ImportError:
    skopt = None

class SkoptParamOptim(ArchOptimBase):
    def __init__(self, space, skopt_args={}):
        super().__init__(space)
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

    def _next(self):
        next_pt = self.skoptim.ask()
        next_params = OrderedDict()
        for n, p in zip(self.param_names, next_pt):
            next_params[n] = p
        return next_params

    # def next(self, batch_size):
    #     next_pts = self.skoptim.ask(n_points=batch_size)
    #     next_params = []
    #     for pt in next_pts:
    #         params = OrderedDict()
    #         for n, p in zip(self.param_names, pt):
    #             params[n] = p
    #         next_params.append(params)
    #     return next_params

    def update(self, estim):
        inputs, results = estim.get_last_results()
        skinputs = [list(inp.values()) for inp in inputs]
        skresults = [-r for r in results]
        self.skoptim.tell(skinputs, skresults)
