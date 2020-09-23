from ..base import MetricsBase
from .. import register, build
from ...arch_space.mixed_ops import MixedOp


@register
class MixedOpTraversalMetrics(MetricsBase):
    def __init__(self, logger, metrics, args={}):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)

    def compute(self, estim):
        mt = 0
        for m in estim.model.mixed_ops():
            for p, op in zip(m.prob(), m.primitives()):
                mt = mt + self.metrics.compute(op) * p
        return mt


@register
class ModuleTraversalMetrics(MetricsBase):
    def __init__(self, logger, metrics, args={}):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)

    def compute(self, estim):
        mt = 0
        for m in estim.model.modules():
            if not isinstance(m, MixedOp):
                mt = mt + self.metrics.compute(m)
            else:
                for p, op in zip(m.prob(), m.primitives()):
                    mt = mt + self.metrics.compute(op) * p
        return mt
