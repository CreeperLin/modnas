import math
import torch
from ..base import MetricsBase
from .. import register, build_metrics

@register('AddAggMetrics')
class AddAggMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, lamd=0.01,):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.lamd = lamd
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model)
        return val + self.lamd * (mt / self.target_val - 1.)


@register('MultAggMetrics')
class MultAggMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model)
        return self.alpha * val * (mt / self.target_val) ** self.beta


@register('MultLogAggMetrics')
class MultLogAggMetrics(MetricsBase):
    def __init__(self, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model)
        return self.alpha * val * (torch.log(mt) / math.log(self.target_val)) ** self.beta


@register('BypassAggMetrics')
class BypassAggMetrics(MetricsBase):
    def __init__(self, metrics, args={}):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)

    def compute(self, val, model):
        mt = self.metrics.compute(model)
        return val
