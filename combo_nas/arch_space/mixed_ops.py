import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.nas_modules import NASModule, ArchModuleSpace
from ..core.param_space import ArchParamDiscrete, ArchParamContinuous
from .ops import build_op
from ..utils import get_current_device
from ..utils.registration import Registry, build, get_builder, register, register_wrapper
from ..utils.profiling import profile_time, report_time
from functools import partial

mixed_op_registry = Registry('mixed_op')
register_mixed_op = partial(register, mixed_op_registry)
get_mixed_op_builder = partial(get_builder, mixed_op_registry)
build_mixed_op = partial(build, mixed_op_registry)
register = partial(register_wrapper, mixed_op_registry)

class DARTSMixedOp(NASModule):
    """ Mixed operation as in DARTS """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None):
        if arch_param_map is None:
            params_shape = (len(ops), )
            arch_param_map = {
                'p': ArchParamContinuous(params_shape),
            }
        super().__init__(arch_param_map)
        self.ops = ops
        self.in_deg = 1 if isinstance(chn_in, int) else len(chn_in)
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_chn_outin[0]
        self.stride = stride
        self._ops = nn.ModuleList()
        for primitive in ops:
            op = build_op(primitive, self.chn_in, self.chn_out, stride)
            self._ops.append(op)
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        p = self.arch_param('p')
        w_path = F.softmax(p.to(device=x.device), dim=-1)
        return sum(w * op(x) for w, op in zip(w_path, self._ops))
    
    def to_genotype(self, k=1):
        ops = self.ops
        w = F.softmax(self.arch_param('p').detach(), dim=-1)
        w_max, prim_idx = torch.topk(w[:-1], 1)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene


class BinGateMixedOp(NASModule):
    """ Mixed operation controlled by binary gate """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None, n_samples=1):
        if arch_param_map is None:
            params_shape = (len(ops), )
            arch_param_map = {
                'p': ArchParamContinuous(params_shape),
            }
        super().__init__(arch_param_map)
        self.ops = ops
        self.in_deg = 1 if isinstance(chn_in, int) else len(chn_in)
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_out[0]
        self.n_samples = n_samples
        self.stride = stride
        self._ops = nn.ModuleList()
        for primitive in ops:
            op = build_op(primitive, self.chn_in, self.chn_out, stride)
            self._ops.append(op)
        self.reset_ops()
        self.s_path_f = None
        # logging.debug("BinGateMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def sample(self):
        p = self.arch_param('p')
        s_op = self.s_op
        w_path = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = w_path.multinomial(self.n_samples)
        self.set_state('w_path_f', w_path, True)
        self.s_path_f = [s_op[i] for i in samples]
    
    def sample_ops(self, n_samples):
        p = self.arch_param('p')
        samples = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())
    
    def reset_ops(self):
        s_op = list(range(len(self._ops)))
        self.last_samples = s_op
        self.s_op = s_op
    
    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        if self.s_path_f is None: self.sample()
        dev_id = ArchModuleSpace.get_dev_id(x.device.index)
        self.set_state('x_f'+dev_id, x.detach())
        s_path_f = self.s_path_f
        if self.training: self.swap_ops(s_path_f)
        self.last_samples = s_path_f
        m_out = sum(self._ops[i](x) for i in s_path_f)
        self.set_state('m_out'+dev_id, m_out)
        return m_out

    def swap_ops(self, samples):
        for i in samples:
            self._ops[i].requires_grad_(True)
        for i in self.last_samples:
            for p in self._ops[i].parameters():
                p.requires_grad = False
                p.grad = None

    def param_grad_dev(self, m_grad, dev_id):
        with torch.no_grad():
            s_op = self.s_op
            a_grad = torch.zeros(self.params_shape)
            m_out = self.get_state('m_out'+dev_id, True)
            if m_out is None: return a_grad
            x_f = self.get_state('x_f'+dev_id)
            w_path_f = self.get_state('w_path_f', True)
            s_path_f = self.s_path_f
            for j, oj in enumerate(s_op):
                if oj in s_path_f:
                    op_out = m_out
                else:
                    op = self._ops[oj].to(device=x_f.device)
                    op_out = op(x_f).detach()
                g_grad = torch.sum(torch.mul(m_grad, op_out)).detach()
                for i, oi in enumerate(s_op):
                    kron = 1 if i==j else 0
                    a_grad[oi] += g_grad * w_path_f[j] * (kron - w_path_f[i])
            a_grad.detach_()
        return a_grad
    
    def to_genotype(self, k=1):
        ops = self.ops
        p = self.arch_param('p')
        w = F.softmax(p.detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, 1)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene


class BinGateUniformMixedOp(BinGateMixedOp):
    """ Mixed operation controlled by binary gate """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None, n_samples=1):
        super().__init__(chn_in, chn_out, stride, ops, arch_param_map, n_samples)
    
    def sample(self):
        p = self.arch_param('p')
        s_op = self.s_op
        w_path = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        self.set_state('w_path_f', w_path)
        prob = F.softmax(torch.empty(w_path.shape, device=w_path.device).uniform_(0, 1), dim=-1)
        samples = prob.multinomial(self.n_samples)
        s_path_f = [s_op[i] for i in samples]
        self.set_state('s_path_f', s_path_f)


class IndexMixedOp(NASModule):
    """ Mixed operation controlled by external index """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None):
        if arch_param_map is None:
            arch_param_map = {
                'ops': ArchParamDiscrete(list(range(len(ops)))),
                # 'ksize': ArchParamDiscrete(['3','5','7']),
            }
        super().__init__(arch_param_map)
        self.ops = ops
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_out[0]
        self.stride = stride
        self._ops = nn.ModuleList([
            build_op(prim, self.chn_in, self.chn_out, stride) for prim in ops
        ])
        self.reset_ops()
        # logging.debug("IndexMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))
        self.chn_out = self.chn_in
    
    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        smp = self.arch_param('ops')
        if self.training: self.swap_ops([smp])
        self.last_samples = [smp]
        return self._ops[smp](x)

    def reset_ops(self):
        s_op = list(range(len(self._ops)))
        self.last_samples = s_op

    def swap_ops(self, samples):
        for i in samples:
            self._ops[i].requires_grad_(True)
        for i in self.last_samples:
            for p in self._ops[i].parameters():
                p.requires_grad = False
                p.grad = None

    def to_genotype(self, k=1):
        return 0, self.ops[self.arch_param('ops')]

class DummyMixedOp(NASModule):
    """ dummy mixed op for non-supernet-based methods """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None):
        if arch_param_map is None:
            arch_param_map = {
                'ops': ArchParamDiscrete(list(range(len(ops)))),
                # 'ksize': ArchParamDiscrete(['3','5','7']),
            }
        super().__init__(arch_param_map)
        self.ops = ops
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_out[0]
        self.stride = stride
        self.chn_out = self.chn_in
    
    def forward(self, x):
        return None

    def to_genotype(self, k=1):
        return 0, self.ops[self.arch_param('ops')]

register_mixed_op(DARTSMixedOp, 'DARTS')
register_mixed_op(BinGateMixedOp, 'BinGate')
register_mixed_op(BinGateUniformMixedOp, 'BinGateUniform')
register_mixed_op(IndexMixedOp, 'Index')
register_mixed_op(DummyMixedOp, 'Dummy')