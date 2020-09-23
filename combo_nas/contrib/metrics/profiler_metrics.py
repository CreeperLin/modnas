import time
import torch
from combo_nas.metrics import register
from combo_nas.metrics.base import MetricsBase


@register
class LocalProfilerMetrics(MetricsBase):
    def __init__(self, logger, device=None, rep=50, warmup=10):
        super().__init__(logger)
        self.rep = rep
        self.warmup = warmup
        self.device = device

    def compute(self, node):
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
                torch.cuda.synchronize()
                op(x)
                torch.cuda.synchronize()
        toc = time.perf_counter()
        lat = 1000. * (toc - tic) / self.rep
        op.to(device=last_device)
        self.logger.debug('local profiler:\tdev: {}\tlat: {:.3f} ms'.format(device, lat))
        return lat
