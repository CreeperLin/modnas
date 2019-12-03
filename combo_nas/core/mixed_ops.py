import torch
import torch.nn as nn
import torch.nn.functional as F
from .nas_modules import NASModule
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
    def __init__(self, chn_in, chn_out, stride, ops, pid=None):
        params_shape = (len(ops), )
        super().__init__(ops, params_shape, pid)
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
    
    def param_forward(self, p):
        w_path = F.softmax(p, dim=-1)
        self.set_state('w_path_f', w_path)
    
    def forward(self, x):
        w_path_f = self.get_state('w_path_f')
        x = x[0] if isinstance(x, list) else x
        return sum(w * op(x) for w, op in zip(w_path_f.to(device=x.device), self._ops))
    
    def to_genotype(self, k=1):
        ops = self.ops
        if self.pid == -1: return -1, [None]
        w = F.softmax(self.arch_param.detach(), dim=-1)
        w_max, prim_idx = torch.topk(w[:-1], 1)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene


class BinGateMixedOp(NASModule):
    """ Mixed operation controlled by binary gate """
    def __init__(self, chn_in, chn_out, stride, ops, pid=None, n_samples=1):
        params_shape = (len(ops), )
        super().__init__(ops, params_shape, pid)
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
        # logging.debug("BinGateMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p, requires_grad=False):
        s_op = self.get_state('s_op')
        w_path = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = w_path.multinomial(self.n_samples)
        self.set_state('w_path_f', w_path, True)
        s_path_f = [s_op[i] for i in samples]
        self.set_state('s_path_f', s_path_f, False)
    
    def sample_ops(self, p, n_samples):
        samples = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        s_op = list(samples.flatten().cpu().numpy())
        self.set_state('s_op', s_op, False)
    
    def reset_ops(self):
        s_op = list(range(len(self._ops)))
        self.last_samples = s_op
        self.set_state('s_op', s_op)
    
    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        dev_id = NASModule.get_dev_id(x.device.index)
        self.set_state('x_f'+dev_id, x.detach())
        smp = self.get_state('s_path_f')
        if self.training: self.swap_ops(smp)
        self.last_samples = smp
        m_out = sum(self._ops[i](x) for i in smp)
        self.set_state('m_out'+dev_id, m_out)
        return m_out

    def swap_ops(self, samples):
        for i in samples:
            self._ops[i].requires_grad_(True)
        for i in self.last_samples:
            for p in self._ops[i].parameters():
                p.requires_grad = False
                p.grad = None

    def param_grad(self, m_grad):
        a_grad = 0
        for dev in NASModule.get_device():
            a_grad = self.param_grad_dev(m_grad, dev) + a_grad
        return a_grad
    
    def param_grad_dev(self, m_grad, dev_id):
        with torch.no_grad():
            sample_ops = self.get_state('s_op')
            a_grad = torch.zeros(self.params_shape)
            m_out = self.get_state('m_out'+dev_id, True)
            if m_out is None: return a_grad
            x_f = self.get_state('x_f'+dev_id)
            w_path_f = self.get_state('w_path_f', True)
            s_path_f = self.get_state('s_path_f', False)
            for j, oj in enumerate(sample_ops):
                if oj in s_path_f:
                    op_out = m_out
                else:
                    op = self._ops[oj].to(device=x_f.device)
                    op_out = op(x_f).detach()
                g_grad = torch.sum(torch.mul(m_grad, op_out)).detach()
                for i, oi in enumerate(sample_ops):
                    kron = 1 if i==j else 0
                    a_grad[oi] += g_grad * w_path_f[j] * (kron - w_path_f[i])
            a_grad.detach_()
        return a_grad
    
    def to_genotype(self, k=1):
        ops = self.ops
        if self.pid == -1: return -1, [None]
        w = F.softmax(self.arch_param.detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, 1)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        return w_max, gene


class BinGateUniformMixedOp(BinGateMixedOp):
    """ Mixed operation controlled by binary gate """
    def __init__(self, chn_in, chn_out, stride, ops, pid=None, n_samples=1):
        super().__init__(chn_in, chn_out, stride, ops, pid=None, n_samples=1)
    
    def param_forward(self, p, requires_grad=False):
        s_op = self.get_state('s_op')
        w_path = F.softmax(p.index_select(-1, s_op), dim=-1)
        self.set_state('w_path_f', w_path)
        prob = F.softmax(torch.empty(w_path.shape, device=w_path.device).uniform_(0, 1), dim=-1)
        samples = prob.multinomial(self.n_samples)
        s_path_f = [s_op[i] for i in samples]
        self.set_state('s_path_f', s_path_f)


class IndexMixedOp(NASModule):
    """ Mixed operation controlled by external index """
    def __init__(self, chn_in, chn_out, stride, ops, pid=None):
        params_shape = (len(ops), )
        super().__init__(ops, params_shape, pid)
        self.ops = ops
        self.in_deg = 1 if isinstance(chn_in, int) else len(chn_in)
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_out[0]
        self.n_samples = n_samples
        self.stride = stride
        self._ops = nn.ModuleList([
            build_op(prim, self.chn_in, self.chn_out, stride) for prim in ops
        ])
        # logging.debug("IndexMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))
        self.params_shape = params_shape
        self.chn_out = self.chn_in
    
    def param_forward(self, p):
        pass

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        smp = self.get_state('s_path_f')
        return sum(self._ops[i](x) for i in smp)

    def to_genotype(self, k=1):
        ops = self.ops
        if self.pid == -1: return -1, [None]
        w_max, prim_idx = torch.topk(w, k)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        return 0, gene


register_mixed_op(DARTSMixedOp, 'DARTS')
register_mixed_op(BinGateMixedOp, 'BinGate')
register_mixed_op(BinGateUniformMixedOp, 'BinGateUniform')