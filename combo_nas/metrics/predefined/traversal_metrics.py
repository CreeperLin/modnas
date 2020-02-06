from ..base import MetricsBase
from .. import register_as, build
from ...arch_space.mixed_ops import MixedOp

@register_as('ModuleMetrics')
class ModuleMetrics(MetricsBase):
    def __init__(self, metrics, args={}):
        super().__init__()
        self.metrics = build(metrics, **args)

    def compute(self, m):
        if not isinstance(m, MixedOp):
            return self.metrics.compute(m)
        mt = 0
        for p, op in zip(m.prob(), m.primitives()):
            mt = mt + self.metrics.compute(op) * p
        return mt


@register_as('MixedOpTraversalMetrics')
class MixedOpTraversalMetrics(MetricsBase):
    def __init__(self, metrics, args={}):
        super().__init__()
        self.metrics = build(metrics, **args)

    def compute(self, model):
        mt = 0
        for m in model.mixed_ops():
            for p, op in zip(m.prob(), m.primitives()):
                mt = mt + self.metrics.compute(op) * p
        return mt


@register_as('ModuleTraversalMetrics')
class ModuleTraversalMetrics(MetricsBase):
    def __init__(self, metrics, args={}):
        super().__init__()
        self.metrics = build(metrics, **args)

    def compute(self, model):
        mt = 0
        for m in model.modules():
            if not isinstance(m, MixedOp):
                mt = mt + self.metrics.compute(m)
            else:
                for p, op in zip(m.prob(), m.primitives()):
                    mt = mt + self.metrics.compute(op) * p
        return mt
