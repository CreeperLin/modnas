import math
import torch
from ..base import MetricsBase
from .. import register, build_metrics

@register('AddLossMetrics')
class AddLossMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, lamd=0.01,):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.lamd = lamd
        self.target_val = float(target_val)

    def compute(self, loss, model):
        mt = self.metrics.compute(model)
        return loss + self.lamd * (mt / self.target_val - 1.)


@register('MultLossMetrics')
class MultLossMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, loss, model):
        mt = self.metrics.compute(model)
        return self.alpha * loss * (mt / self.target_val) ** self.beta


@register('MultLogLossMetrics')
class MultLogLossMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, loss, model):
        mt = self.metrics.compute(model)
        return self.alpha * loss * (torch.log(mt) / math.log(self.target_val)) ** self.beta


@register('BypassLossMetrics')
class BypassLossMetrics(MetricsBase):
    def __init__(self, metrics, args={}):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)

    def compute(self, loss, model):
        mt = self.metrics.compute(model)
        return loss
