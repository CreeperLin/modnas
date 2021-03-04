from modnas.utils import format_value
from modnas.registry.callback import register
from modnas.callback.base import CallbackBase


def MIN_CMP(x, y):
    return 0 if x is None or y is None else y - x


def MAX_CMP(x, y):
    return 0 if x is None or y is None else x - y


@register
class OptimumReporter(CallbackBase):

    priority = 0

    def __init__(self, cmp_keys=None, cmp_fn=None, cmp_th=None, score_fn=None, stat_epoch=True):
        handlers = {
            'after:EstimBase.step': self.on_step,
            'after:EstimBase.run': self.report_results,
        }
        if stat_epoch:
            handlers['after:EstimBase.run_epoch'] = self.report_epoch
        super().__init__(handlers)
        self.stat_epoch = stat_epoch
        self.cmp_keys = cmp_keys
        cmp_fn = cmp_fn or {}
        cmp_fn = {k: (MAX_CMP if v == 'max' else MIN_CMP if v == 'min' else v) for k, v in cmp_fn.items()}
        self.cmp_fn = cmp_fn or {}
        self.cmp_th = cmp_th or {}
        self.score_fn = score_fn
        self.results = []
        self.opt_results = []
        self.ep_opt_results = []

    def update_optimal(self, res, opts):
        met = res[1]
        if self.cmp_keys is None:
            self.cmp_keys = list(met.keys())
        rem_opt = []
        for i, (_, m) in enumerate(opts):
            c = self.dom_cmp(met, m)
            if c < 0:
                return opts
            elif c > 0:
                rem_opt.append(i)
        opts = [r for i, r in enumerate(opts) if i not in rem_opt]
        opts.append(res)
        return opts

    def dom_cmp(self, m1, m2):
        dom = 0
        for k in self.cmp_keys:
            v1, v2 = m1.get(k, None), m2.get(k, None)
            cmp = self.cmp_fn.get(k, MAX_CMP)(v1, v2)
            th = self.cmp_th.get(k, 0)
            if cmp > th:
                if dom == -1:
                    return 0
                dom = 1
            elif cmp < -th:
                if dom == 1:
                    return 0
                dom = -1
        return dom

    def on_step(self, ret, estim, params):
        if self.score_fn:
            ret = {'score': self.score_fn(ret)}
        if not isinstance(ret, dict):
            ret = {'default': ret}
        res = (dict(params), ret)
        self.results.append(res)
        self.opt_results = self.update_optimal(res, self.opt_results)
        if self.stat_epoch:
            self.ep_opt_results = self.update_optimal(res, self.ep_opt_results)

    def format_metrics(self, opts):
        if not opts:
            return None
        met = [r[1] for r in opts]
        met = [{k: format_value(v, unit=False, factor=0, prec=4) for k, v in m.items()} for m in met]
        if len(met[0]) == 1:
            met[0] = list(met[0].values())[0]
        if len(met) == 1:
            met = met[0]
        return met

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        estim.stats['epoch_opt'] = self.format_metrics(self.ep_opt_results)
        estim.stats['opt'] = self.format_metrics(self.opt_results)
        self.ep_opt_results = []

    def report_results(self, ret, estim, optim):
        opt_res = {
            'opt_results': self.opt_results,
        }
        ret = ret or {}
        ret.update(opt_res)
        return ret
