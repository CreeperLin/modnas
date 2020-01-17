import time
import torch
from combo_nas import metrics
from combo_nas.metrics.base import MetricsBase

@metrics.register('LocalProfilerMetrics')
class LocalProfilerMetrics(MetricsBase):
    def __init__(self, device=None, head=None, rep=50):
        super().__init__()
        if head is None:
            head = ['name']
        self.results = {}
        self.rep = rep
        self.head = head
        self.device = device

    def compute(self, node):
        key = '#'.join([str(node[k]) for k in self.head])
        if key in self.results:
            return self.results[key]
        in_shape = node['in_shape']
        op = node.module
        plist = list(op.parameters())
        if len(plist) == 0:
            last_device = None
        else:
            last_device = plist[0].device
        device = last_device if self.device is None else self.device
        x = torch.randn(in_shape).to(device=device)
        op = op.to(device=device)
        tic = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.rep):
                op(x)
        toc = time.perf_counter()
        lat = 1000. * (toc - tic) / self.rep
        self.results[key] = lat
        op.to(device=last_device)
        print('measure: {} {} {} {:.3f}'.format(key, in_shape, device, lat))
        return lat
        