import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.param_space import ArchParamCategorical, ArchParamTensor
from .ops import build_op
from ..utils.registration import Registry, build, get_builder, register, register_wrapper

mixed_op_registry = Registry('mixed_op')
register_mixed_op = partial(register, mixed_op_registry)
get_mixed_op_builder = partial(get_builder, mixed_op_registry)
build_mixed_op = partial(build, mixed_op_registry)
register = partial(register_wrapper, mixed_op_registry)

class MixedOp(nn.Module):
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map):
        super().__init__()
        self.arch_param_map = arch_param_map
        for ap in arch_param_map.values():
            ap.add_module(self)
        self.ops = ops
        self.in_deg = 1 if isinstance(chn_in, int) else len(chn_in)
        self.chn_in = chn_in if isinstance(chn_in, int) else chn_in[0]
        self.chn_out = chn_out if isinstance(chn_out, int) else chn_in[0]
        self.stride = stride
        self.w_max = 0
        self._ops = nn.ModuleList([
            build_op(prim, self.chn_in, self.chn_out, stride) for prim in ops
        ])
        logging.debug('mixed op: {} p: {}'.format(type(self), arch_param_map))

    def primitives(self):
        return self._ops

    def alpha(self):
        return self.arch_param_value('p')

    def prob(self):
        return F.softmax(self.alpha(), dim=-1)

    def arch_param(self, name):
        return self.arch_param_map.get(name)

    def arch_param_value(self, name):
        return self.arch_param_map.get(name).value()

    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError


class WeightedSumMixedOp(MixedOp):
    """ Mixed operation as weighted sum """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None):
        if arch_param_map is None:
            params_shape = (len(ops), )
            arch_param_map = {
                'p': ArchParamTensor(params_shape),
            }
        super().__init__(chn_in, chn_out, stride, ops, arch_param_map)

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        p = self.arch_param_value('p')
        w_path = F.softmax(p.to(device=x.device), dim=-1)
        return sum(w * op(x) for w, op in zip(w_path, self._ops))

    def to_genotype(self, k=1):
        ops = self.ops
        w = F.softmax(self.arch_param_value('p').detach(), dim=-1)
        # ignore last primitive (None)
        w_max, prim_idx = torch.topk(w[:-1], k)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        self.w_max = w_max
        return gene


class BinGateMixedOp(MixedOp):
    """ Mixed operation controlled by binary gate """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None, n_samples=1):
        if arch_param_map is None:
            params_shape = (len(ops), )
            arch_param_map = {
                'p': ArchParamTensor(params_shape),
            }
        super().__init__(chn_in, chn_out, stride, ops, arch_param_map)
        self.n_samples = n_samples
        self.reset_ops()
        self.s_path_f = None
        self.a_grad_enabled = False
        # logging.debug("BinGateMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))

    def arch_param_grad(self, enabled):
        self.a_grad_enabled = enabled

    def sample_path(self):
        p = self.arch_param_value('p')
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = self.w_path_f.multinomial(self.n_samples)
        self.s_path_f = [s_op[i] for i in samples]

    def sample_ops(self, n_samples):
        p = self.arch_param_value('p')
        samples = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())

    def reset_ops(self):
        s_op = list(range(len(self._ops)))
        self.last_samples = s_op
        self.s_op = s_op

    def forward(self, x):
        self.sample_path()
        x = x[0] if isinstance(x, list) else x

        s_path_f = self.s_path_f
        ops = self._ops
        if self.training: 
            self.swap_ops(s_path_f)
        if self.a_grad_enabled:
            p = self.arch_param_value('p')
            ctx_dict = {
                's_op': self.s_op,
                's_path_f': self.s_path_f,
                'w_path_f': self.w_path_f,
                'ops': ops,
            }
            m_out = BinGateFunction.apply(x, p, ctx_dict)
        else:
            if len(s_path_f) == 1: m_out = ops[s_path_f[0]](x)
            else: m_out = sum(ops[i](x) for i in s_path_f)
        self.last_samples = s_path_f
        return m_out

    def swap_ops(self, samples):
        for i in samples:
            for p in self._ops[i].parameters():
                p.requires_grad_(True)
        for i in self.last_samples:
            for p in self._ops[i].parameters():
                p.requires_grad = False
                p.grad = None

    def to_genotype(self, k=1):
        ops = self.ops
        p = self.arch_param_value('p')
        w = F.softmax(p.detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, k)
        gene = [ops[i] for i in prim_idx]
        if gene == []: return -1, [None]
        self.w_max = w_max
        return gene


class BinGateFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(ctx, x, alpha, ctx_dict):
        ctx.__dict__.update(ctx_dict)
        ctx.param_shape = alpha.shape
        ops = ctx.ops
        s_path_f = ctx.s_path_f
        with torch.enable_grad():
            if len(s_path_f) == 1: m_out = ops[s_path_f[0]](x)
            else: m_out = sum(ops[i](x) for i in s_path_f)
        ctx.save_for_backward(x, m_out)
        return m_out.data

    @staticmethod
    def backward(ctx, m_grad):
        x_f, m_out = ctx.saved_tensors
        grad_x = torch.autograd.grad(m_out, x_f, m_grad, only_inputs=True)
        with torch.no_grad():
            a_grad = torch.zeros(ctx.param_shape)
            s_op = ctx.s_op
            w_path_f = ctx.w_path_f
            s_path_f = ctx.s_path_f
            ops = ctx.ops
            for j, oj in enumerate(s_op):
                if oj in s_path_f:
                    op_out = m_out.data
                else:
                    op = ops[oj]
                    op_out = op(x_f.data)
                g_grad = torch.sum(m_grad * op_out)
                for i, oi in enumerate(s_op):
                    kron = 1 if i==j else 0
                    a_grad[oi] += g_grad * w_path_f[j] * (kron - w_path_f[i])
        return grad_x[0], a_grad, None


class BinGateUniformMixedOp(BinGateMixedOp):
    """ Mixed operation controlled by binary gate """
    def sample_path(self):
        p = self.arch_param_value('p')
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        # sample uniformly
        samples = F.softmax(torch.ones(len(s_op)), dim=-1).multinomial(self.n_samples)
        s_path_f = [s_op[i] for i in samples]
        self.s_path_f = s_path_f

    def sample_ops(self, n_samples):
        p = self.arch_param_value('p')
        # sample uniformly
        samples = F.softmax(torch.ones(p.shape), dim=-1).multinomial(n_samples).detach()
        # sample according to p
        # samples = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())


class IndexMixedOp(MixedOp):
    """ Mixed operation controlled by external index """
    def __init__(self, chn_in, chn_out, stride, ops, arch_param_map=None):
        if arch_param_map is None:
            arch_param_map = {
                'ops': ArchParamCategorical(ops),
            }
        super().__init__(chn_in, chn_out, stride, ops, arch_param_map)
        self.reset_ops()
        # logging.debug("IndexMixedOp: chn_in:{} stride:{} #p:{:.6f}".format(self.chn_in, stride, param_count(self)))

    def alpha(self):
        alpha = torch.zeros(len(self.primitives()))
        alpha[self.arch_param('ops').index()] = 1.0
        return alpha

    def prob(self):
        return self.alpha()

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        smp = self.arch_param('ops').index()
        if self.training: self.swap_ops([smp])
        self.last_samples = [smp]
        return self._ops[smp](x)

    def reset_ops(self):
        s_op = list(range(len(self._ops)))
        self.last_samples = s_op

    def swap_ops(self, samples):
        for i in samples:
            for p in self._ops[i].parameters():
                p.requires_grad_(True)
        for i in self.last_samples:
            for p in self._ops[i].parameters():
                p.requires_grad = False
                p.grad = None

    def to_genotype(self, *args, **kwargs):
        return 0, self.arch_param_value('ops')


register_mixed_op(WeightedSumMixedOp, 'WeightedSum')
register_mixed_op(BinGateMixedOp, 'BinGate')
register_mixed_op(BinGateUniformMixedOp, 'BinGateUniform')
register_mixed_op(IndexMixedOp, 'Index')
