from ..base import MetricsBase
from .. import register_as, build
from ...arch_space.constructor import Slot
try:
    import rasp
    import rasp.frontend as F
except ImportError:
    rasp = None

@register_as('RASPStatsMetrics')
class RASPStatsMetrics(MetricsBase):
    def __init__(self, logger, item):
        super().__init__(logger)
        self.item = item

    def compute(self, node):
        return node[self.item]


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
        return mt


@register_as('RASPTraversalMetrics')
class RASPTraversalMetrics(MetricsBase):
    def __init__(self, logger, input_shape, metrics, args={}, compute=True, timing=False,
                device=None, mixed_only=False):
        super().__init__(logger)
        if rasp is None: raise ValueError('package RASP is not found')
        self.metrics = build(metrics, logger, **args)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device
        self.mixed_only = mixed_only
        self.excluded = set()

    def compute_tape_recursive(self, node):
        if node in self.excluded:
            return 0
        ret = self.metrics.compute(node)
        if not ret is None:
            return ret
        ret = 0
        for n in node.tape.items:
            module = n.module
            if isinstance(module, Slot):
                ent_node = F.get_stats_node(module.ent)
                prim_type = module.gene
                if isinstance(prim_type, (tuple, list)):
                    prim_type = prim_type[0]
                ent_node['prim_type'] = prim_type
            n_ret = self.compute_tape_recursive(n)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

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
        for m in model.mixed_ops():
            mixop_node = F.get_stats_node(m)
            self.excluded.add(mixop_node)
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
        if not self.mixed_only:
            mt = mt + self.compute_tape_recursive(root)
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
