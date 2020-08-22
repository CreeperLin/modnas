import torch
from ..base import MetricsBase
from .. import register_as, build
from ...arch_space.slot import Slot
from ...arch_space.mixed_ops import MixedOp
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


@register_as('RASPTraversalMetrics')
class RASPTraversalMetrics(MetricsBase):
    def __init__(self, logger, input_shape, metrics, args={}, compute=True, timing=False,
                device='cuda', mixed_only=False, keep_stats=True, traversal_type='tape_nodes'):
        super().__init__(logger)
        if rasp is None: raise ValueError('package RASP is not found')
        self.metrics = build(metrics, logger, **args)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device
        self.mixed_only = mixed_only
        self.keep_stats = keep_stats
        if traversal_type == 'tape_leaves':
            self.traverse = self.traverse_tape_leaves
        elif traversal_type == 'tape_nodes':
            self.traverse = self.traverse_tape_nodes
        else:
            raise ValueError('invalid traversal type')
        self.excluded = set()

    def traverse_tape_nodes(self, node):
        ret = 0
        if node in self.excluded:
            return ret
        if node.num_children == 0:
            ret = self.metrics.compute(node)
            return ret
        if node.tape is None:
            return ret
        for cur_node in node.tape.items:
            module = cur_node.module
            if isinstance(module, Slot):
                prim_type = module.desc
                if not prim_type is None:
                    if isinstance(prim_type, (tuple, list)):
                        prim_type = prim_type[0]
                    ent_node = F.get_stats_node(module.ent)
                    ent_node['prim_type'] = prim_type
            if cur_node['prim_type']:
                n_ret = self.metrics.compute(cur_node)
            else:
                n_ret = self.traverse_tape_nodes(cur_node)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

    def traverse_tape_leaves(self, node):
        ret = 0
        for cur_node in node.tape.items_all:
            if cur_node in self.excluded:
                continue
            n_ret = self.metrics.compute(cur_node)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

    def stat(self, module, input_shape):
        if self.eval_compute:
            F.hook_compute(module)
        if self.eval_timing:
            F.hook_timing(module)
        F.run(module, F.get_random_data(input_shape), self.device)
        F.unhook_compute(module)
        F.unhook_timing(module)

    def compute(self, net):
        self.excluded.clear()
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            self.stat(net, self.input_shape)
        mt = 0
        for m in net.modules():
            if not isinstance(m, MixedOp):
                continue
            mixop_node = F.get_stats_node(m)
            self.excluded.add(mixop_node)
            assert mixop_node['in_shape'] is not None
            mixop_mt = 0
            m_in, m_out = mixop_node['in_shape'], mixop_node['out_shape']
            for p, (pn, op) in zip(m.prob(), m.named_primitives()):
                if not p:
                    continue
                subn = F.get_stats_node(op)
                if subn['prim_type'] is None:
                    subn['prim_type'] = pn
                if subn['compute_updated'] is None:
                    if subn['in_shape'] is None:
                        subn['in_shape'] = m_in
                    self.stat(subn.module, subn['in_shape'])
                    subn['compute_updated'] = True
                subn_mt = self.metrics.compute(subn)
                if subn_mt is None:
                    subn_mt = self.traverse(subn)
                if subn_mt is None:
                    self.logger.warning('unresolved node: {} type: {}'.format(node['name'], node['type']))
                    subn_mt = 0
                mixop_mt = mixop_mt + subn_mt * p
            mt += mixop_mt
        if not self.mixed_only:
            mt = mt + self.traverse(root)
        if not self.keep_stats:
            F.unreg_stats_node(net)
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

    def compute(self, net):
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
