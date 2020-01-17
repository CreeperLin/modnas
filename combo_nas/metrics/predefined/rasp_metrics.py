from ..base import MetricsBase
from .. import register, build_metrics
try:
    import rasp
except ImportError:
    rasp = None

@register('RASPLatencyMetrics')
class RASPLatencyMetrics(MetricsBase):
    def __init__(self):
        super().__init__()

    def compute(self, node):
        return 0 if node['lat'] is None else node['lat']


@register('RASPFLOPSMetrics')
class RASPFLOPSMetrics(MetricsBase):
    def __init__(self):
        super().__init__()

    def compute(self, node):
        return 0 if node['flops'] is None else node['flops']


@register('RASPStatsDelegateMetrics')
class RASPStatsDelegateMetrics(MetricsBase):
    def __init__(self, metrics, args={}, ignore_none=True):
        super().__init__()
        self.metrics = build_metrics(metrics, **args)
        self.ignore_none = ignore_none

    def compute(self, node):
        stats = node.stats
        mt = self.metrics.compute(stats)
        if not self.ignore_none and mt is None:
            raise ValueError('Metrics return None for input: {}'.format(stats))
        return 0 if mt is None else mt


@register('RASPTraversalMetrics')
class RASPTraversalMetrics(MetricsBase):
    def __init__(self, input_shape, metrics, args={}, compute=True, timing=False):
        super().__init__()
        if rasp is None: raise ValueError('package RASP is not found')
        self.metrics = build_metrics(metrics, **args)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape

    def compute(self, model):
        net = model.net
        dev_ids = model.device_ids
        dev_id = 'cpu' if len(dev_ids) == 0 else dev_ids[0]
        if not hasattr(net, '_RASPStatNode'):
            F = rasp.frontend
            root = F.reg_stats_node(net)
            if self.eval_compute:
                F.hook_compute(net)
            if self.eval_timing:
                F.hook_timing(net)
            inputs = F.get_random_data(self.input_shape).to(dev_id)
            F.run(net, inputs)
            F.unhook_compute(net)
            F.unhook_timing(net)
        else:
            root = net._RASPStatNode
        mt = 0
        for node in root.tape.items_all:
            if '_ops' in node.name:
                continue
            mt = mt + self.metrics.compute(node)
        for m in model.mixed_ops():
            mixop_node = m._RASPStatNode
            assert mixop_node['in_shape'] is not None
            mixop_mt = 0
            m_in, m_out = mixop_node['in_shape'], mixop_node['out_shape']
            for p, (pn, op) in zip(m.prob(), m.named_primitives()):
                subn = op._RASPStatNode
                if subn['prim_type'] is None:
                    subn['prim_type'] = pn
                if subn['compute_updated'] is None:
                    rasp.profiler.eval.eval_compute_nofwd(subn, m_in, m_out)
                smt = self.metrics.compute(subn)
                mixop_mt = mixop_mt + smt * p.to(device=dev_id)
            mt += mixop_mt
        # print('current flops: {}'.format(mt.item()))
        return mt
