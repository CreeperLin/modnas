import yaml
import pickle
import numpy as np
from ..base import MetricsBase
from .. import register_as

@register_as('StatsLUTMetrics')
class StatsLUTMetrics(MetricsBase):
    def __init__(self, logger, lut_path, head=None):
        super().__init__(logger)
        with open(lut_path, 'r') as f:
            self.lut = yaml.load(f, Loader=yaml.Loader)
        if self.lut is None:
            raise ValueError('StatsLUTMetrics: Error loading LUT: {}'.format(lut_path))
        self.head = head
        self.warned = set()

    def compute(self, stats):
        key = '#'.join([str(stats[k]) for k in self.head if not stats.get(k, None) is None])
        val = self.lut.get(key, None)
        if val is None:
            if not key in self.warned:
                self.logger.warning('StatsLUTMetrics: missing key in LUT: {}'.format(key))
                self.warned.add(key)
        elif isinstance(val, dict):
            val = float(np.random.normal(val['mean'], val['std']))
        else:
            val = float(val)
        return val


@register_as('StatsModelMetrics')
class StatsModelMetrics(MetricsBase):
    def __init__(self, logger, model_path, head):
        super().__init__(logger)
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.head = head

    def compute(self, stats):
        feats = [stats.get(c, None) for c in self.head]
        return self.model.predict(feats)
