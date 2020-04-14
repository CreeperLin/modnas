import time
import torch
from combo_nas import metrics
from combo_nas.metrics.base import MetricsBase

@metrics.register_as('LocalProfilerMetrics')
class LocalProfilerMetrics(MetricsBase):
    def __init__(self, logger, device=None, head=None, rep=50, warmup=10):
        super().__init__(logger)
        if head is None:
            head = ['name']
        self.results = {}
        self.rep = rep
        self.warmup = warmup
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
            for rep in range(self.warmup + self.rep):
                if rep == self.warmup:
                    tic = time.perf_counter()
                torch.cuda.synchronize(device=device)
                op(x)
                torch.cuda.synchronize(device=device)
        toc = time.perf_counter()
        lat = 1000. * (toc - tic) / self.rep
        self.results[key] = lat
        op.to(device=last_device)
        self.logger.info('local profiler: {}\tdev: {}\tlat: {:.3f} ms'.format(key, device, lat))
        return lat
        