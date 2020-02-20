import math
import torch
from ..base import MetricsBase
from .. import register_as, build

@register_as('AddAggMetrics')
class AddAggMetrics(MetricsBase):
    def __init__(self, logger, target_val, metrics, args={}, lamd=0.01,):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)
        self.lamd = lamd
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model).to(device=val.device)
        return val + self.lamd * (mt / self.target_val - 1.)


@register_as('MultAggMetrics')
class MultAggMetrics(MetricsBase):
    def __init__(self, logger, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model).to(device=val.device)
        return self.alpha * val * (mt / self.target_val) ** self.beta


@register_as('MultLogAggMetrics')
class MultLogAggMetrics(MetricsBase):
    def __init__(self, logger, target_val, metrics, args={}, alpha=1., beta=0.6):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)

    def compute(self, val, model):
        mt = self.metrics.compute(model).to(device=val.device)
        return self.alpha * val * (torch.log(mt) / math.log(self.target_val)) ** self.beta


@register_as('BypassAggMetrics')
class BypassAggMetrics(MetricsBase):
    def __init__(self, logger, metrics, args={}):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)

    def compute(self, val, model):
        mt = self.metrics.compute(model)
        return val
