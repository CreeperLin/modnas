import yaml
import pickle
from ..base import MetricsBase
from .. import register_as

@register_as('StatsLUTMetrics')
class StatsLUTMetrics(MetricsBase):
    def __init__(self, lut_path, head=None):
        super().__init__()
        with open(lut_path, 'r') as f:
            self.lut = yaml.load(f, Loader=yaml.Loader)
        if self.lut is None:
            raise ValueError('StatsLUTMetrics: Error loading LUT: {}'.format(lut_path))
        self.head = head

    def compute(self, stats):
        keys = []
        for h in self.head:
            head = stats.get(h, None)
            if head is not None:
                keys.append(str(head))
        key = '#'.join(keys)
        val = self.lut.get(key, None)
        if val is None:
            print('StatsLUTMetrics: missing key in LUT: {}'.format(key))
        return val


@register_as('StatsModelMetrics')
class StatsModelMetrics(MetricsBase):
    def __init__(self, model_path, head):
        super().__init__()
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.head = head

    def compute(self, stats):
        feats = [stats.get(c, None) for c in self.head]
        return self.model.predict(feats)
