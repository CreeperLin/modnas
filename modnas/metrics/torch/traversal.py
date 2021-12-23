"""Network module traversal metrics."""
from ..base import MetricsBase
from modnas.registry.metrics import register, build
from modnas.arch_space.torch.mixed_ops import MixedOp


@register
class MixedOpTraversalMetrics(MetricsBase):
    """Mixed operator traversal metrics class."""

    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        """Return metrics output."""
        mt = 0
        for m in estim.model.mixed_ops():
            for p, op in zip(m.prob(), m.candidates()):
                mt = mt + self.metrics(op) * p
        return mt


@register
class ModuleTraversalMetrics(MetricsBase):
    """Module traversal metrics class."""

    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        """Return metrics output."""
        mt = 0
        for m in estim.model.modules():
            if not isinstance(m, MixedOp):
                mt = mt + self.metrics(m)
            else:
                for p, op in zip(m.prob(), m.candidates()):
                    mt = mt + self.metrics(op) * p
        return mt
