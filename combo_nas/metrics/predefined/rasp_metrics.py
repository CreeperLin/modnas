from ..base import MetricsBase
from .. import register, build_metrics
from ...arch_space.mixed_ops import MixedOp
try:
    import rasp.torch as rasptorch
except ImportError:
    rasptorch = None

@register('RASPLatencyMetrics')
class RASPLatencyMetrics(MetricsBase):
    def __init__(self):
        super().__init__()
        if rasptorch is None: raise ValueError('package RASP is not found')

    def compute(self, node):
        return 0 if node['lat'] is None else node['lat']


@register('RASPDeviceLatencyMetrics')
class RASPDeviceLatencyMetrics(MetricsBase):
    def __init__(self):
        super().__init__()
        if rasptorch is None: raise ValueError('package RASP is not found')

    def compute(self, node):
        return 0 if node['lat'] is None else node['lat']


@register('RASPFLOPSMetrics')
class RASPFLOPSMetrics(MetricsBase):
    def __init__(self):
        super().__init__()
        if rasptorch is None: raise ValueError('package RASP is not found')

    def compute(self, node):
        return 0 if node['flops'] is None else node['flops']


@register('RASPTraversalMetrics')
class RASPTraversalMetrics(MetricsBase):
    def __init__(self, metrics, args={}, compute=True, timing=False):
        super().__init__()
        if rasptorch is None: raise ValueError('package RASP is not found')
        self.metrics = build_metrics(metrics, **args)
        self.eval_compute = compute
        self.eval_timing = timing

    def compute(self, model):
        input_shape = model.input_shape()
        dev_id = model.device_ids[0]
        if input_shape is None:
            return 0
        if not hasattr(model, '_RASPStatNode'):
            root = rasptorch.frontend.reg_stats_node(model)
            if self.eval_compute:
                rasptorch.frontend.hook_compute(model)
            if self.eval_timing:
                rasptorch.frontend.hook_timing(model)
            prof_input_shape = (1, ) + input_shape[1:]
            inputs = rasptorch.frontend.get_random_data(prof_input_shape).to(dev_id)
            rasptorch.frontend.run(model, inputs)
            rasptorch.frontend.unhook_compute(model)
            rasptorch.frontend.unhook_timing(model)
        else:
            root = model._RASPStatNode
        mt = 0
        for node in root.tape.items_all:
            if '_ops' in node.name:
                continue
            mt = mt + self.metrics.compute(node)
            print(node.name, self.metrics.compute(node))
        print(mt)
        for m in model.mixed_ops():
            mixop_node = m._RASPStatNode
            assert mixop_node['in_shape'] is not None
            for p, op in zip(m.prob(), m.primitives()):
                subn = op._RASPStatNode
                if subn['compute_updated'] is None:
                    rasptorch.eval_compute_nofwd(subn, mixop_node['in_shape'], mixop_node['out_shape'])
                mt = mt + self.metrics.compute(subn) * p.to(device=dev_id)
        print('current flops: {}'.format(mt.item()))
        return mt
