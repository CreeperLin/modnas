from ..base import MetricsBase
from .. import register_as, build

@register_as('SumAgg')
class SumAggMetrics(MetricsBase):
    def __init__(self, logger, metrics_conf):
        super().__init__(logger)
        self.metrics = {k: build(conf.type, logger, **conf.get('args', {}))
                       for k, conf in metrics_conf.items()}
        self.base = {k: conf.get('base', 1) for k, conf in metrics_conf.items()}
        self.weight = {k: conf.get('weight', 1) for k, conf in metrics_conf.items()}

    def compute(self, item):
        mt_res = {k: (mt.compute(item) or 0) for k, mt in self.metrics.items()}
        self.logger.info('SumAgg: {{{}}}'.format(', '.join(['{}: {}'.format(k, r) for k, r in mt_res.items()])))
        return sum(self.weight[k] * mt_res[k] / self.base[k] for k in self.metrics)


@register_as('MergeAgg')
class MergeAggMetrics(MetricsBase):
    def __init__(self, logger, metrics_conf):
        super().__init__(logger)
        self.metrics = {k: build(conf.type, logger, **conf.get('args', {}))
                        for k, conf in metrics_conf.items()}

    def compute(self, item):
        return {k: mt.compute(item) for k, mt in self.metrics.items()}
