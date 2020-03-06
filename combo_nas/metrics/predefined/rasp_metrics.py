from ..base import MetricsBase
from .. import register_as, build
try:
    import rasp
    import rasp.frontend as F
except ImportError:
    rasp = None

@register_as('RASPLatencyMetrics')
class RASPLatencyMetrics(MetricsBase):
    def __init__(self, logger):
        super().__init__(logger)

    def compute(self, node):
        return 0 if node['lat'] is None else node['lat']


@register_as('RASPFLOPSMetrics')
class RASPFLOPSMetrics(MetricsBase):
    def __init__(self, logger):
        super().__init__(logger)

    def compute(self, node):
        return 0 if node['flops'] is None else node['flops']


@register_as('RASPStatsDelegateMetrics')
class RASPStatsDelegateMetrics(MetricsBase):
    def __init__(self, logger, metrics, args={}, ignore_none=True):
        super().__init__(logger)
        self.metrics = build(metrics, logger, **args)
        self.ignore_none = ignore_none

    def compute(self, node):
        stats = node.stats
        mt = self.metrics.compute(stats)
        if not self.ignore_none and mt is None:
            raise ValueError('Metrics return None for input: {}'.format(stats))
        return 0 if mt is None else mt


@register_as('RASPTraversalMetrics')
class RASPTraversalMetrics(MetricsBase):
    def __init__(self, logger, input_shape, metrics, args={}, compute=True, timing=False, device=None):
        super().__init__(logger)
        if rasp is None: raise ValueError('package RASP is not found')
        self.metrics = build(metrics, logger, **args)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device

    def compute(self, model):
        net = model.net
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            if self.eval_compute:
                F.hook_compute(net)
            if self.eval_timing:
                F.hook_timing(net)
            inputs = F.get_random_data(self.input_shape)
            F.run(net, inputs, self.device)
            F.unhook_compute(net)
            F.unhook_timing(net)
        mt = 0
        for node in root.tape.items_all:
            if '_ops' in node.name:
                continue
            mt = mt + self.metrics.compute(node)
        for m in model.mixed_ops():
            mixop_node = F.get_stats_node(m)
            assert mixop_node['in_shape'] is not None
            mixop_mt = 0
            m_in, m_out = mixop_node['in_shape'], mixop_node['out_shape']
            for p, (pn, op) in zip(m.prob(), m.named_primitives()):
                subn = F.get_stats_node(op)
                if subn['prim_type'] is None:
                    subn['prim_type'] = pn
                if subn['compute_updated'] is None:
                    rasp.profiler.eval.eval_compute_nofwd(subn, m_in, m_out)
                smt = self.metrics.compute(subn)
                mixop_mt = mixop_mt + smt * p
            mt += mixop_mt
        return mt


@register_as('RASPRootMetrics')
class RASPRootMetrics(MetricsBase):
    def __init__(self, logger, input_shape, metrics, args={}, compute=True, timing=False, device=None):
        super().__init__(logger)
        if rasp is None: raise ValueError('package RASP is not found')
        self.metrics = build(metrics, logger, **args)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device

    def compute(self, model):
        net = model.net
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            if self.eval_compute:
                F.hook_compute(net)
            if self.eval_timing:
                F.hook_timing(net)
            inputs = F.get_random_data(self.input_shape)
            F.run(net, inputs, self.device)
            F.unhook_compute(net)
            F.unhook_timing(net)
        return self.metrics.compute(root)
