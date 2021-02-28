from ..base import MetricsBase
from modnas.registry.metrics import register, build
from ...arch_space.mixed_ops import MixedOp


@register
class MixedOpTraversalMetrics(MetricsBase):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        mt = 0
        for m in estim.model.mixed_ops():
            for p, op in zip(m.prob(), m.primitives()):
                mt = mt + self.metrics(op) * p
        return mt


@register
class ModuleTraversalMetrics(MetricsBase):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        mt = 0
        for m in estim.model.modules():
            if not isinstance(m, MixedOp):
                mt = mt + self.metrics(m)
            else:
                for p, op in zip(m.prob(), m.primitives()):
                    mt = mt + self.metrics(op) * p
        return mt
