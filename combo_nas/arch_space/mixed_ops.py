import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.param_space import ArchParamCategorical, ArchParamTensor
from ..utils.registration import get_registry_utils

from . import register

class MixedOp(nn.Module):
    def __init__(self, primitives, arch_param_map):
        super().__init__()
        if isinstance(primitives, (tuple, list)):
            primitives = {n: p for n, p in primitives}
        if isinstance(primitives, dict):
            self._ops = nn.ModuleDict(primitives)
        else:
            raise ValueError('unsupported primitives type')
        if arch_param_map is None:
            arch_param_map = {
                'p': ArchParamTensor(len(self._ops)),
            }
        self.arch_param_map = arch_param_map
        for ap in arch_param_map.values():
            ap.add_module(self)
        logging.debug('mixed op: {} p: {}'.format(type(self), arch_param_map))

    def primitives(self):
        return list(self._ops.values())

    def primitive_names(self):
        return list(self._ops.keys())

    def named_primitives(self):
        for n, prim in self._ops.items():
            yield n, prim

    def alpha(self):
        return self.arch_param_value('p')

    def prob(self):
        return F.softmax(self.alpha(), dim=-1)

    def arch_param(self, name):
        return self.arch_param_map.get(name)

    def arch_param_value(self, name):
        return self.arch_param_map.get(name).value()

    def to_arch_desc(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def gen(model):
        for m in model.modules():
            if isinstance(m, MixedOp):
                yield m


class WeightedSumMixedOp(MixedOp):
    """ Mixed operation as weighted sum """
    def __init__(self, primitives, arch_param_map=None):
        super().__init__(primitives, arch_param_map)

    def forward(self, *args, **kwargs):
        outputs = [op(*args, **kwargs) for op in self.primitives()]
        w_path = F.softmax(self.alpha().to(device=outputs[0].device), dim=-1)
        return sum(w * o for w, o in zip(w_path, outputs))

    def to_arch_desc(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []: return [None]
        return desc


class BinGateMixedOp(MixedOp):
    """ Mixed operation controlled by binary gate """
    def __init__(self, primitives, arch_param_map=None, n_samples=1):
        super().__init__(primitives, arch_param_map)
        self.n_samples = n_samples
        self.s_path_f = None
        self.last_samples = []
        self.s_op = []
        self.a_grad_enabled = False
        self.reset_ops()

    def arch_param_grad(self, enabled):
        self.a_grad_enabled = enabled

    def sample_path(self):
        p = self.arch_param_value('p')
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = self.w_path_f.multinomial(1 if self.a_grad_enabled else self.n_samples)
        self.s_path_f = [s_op[i] for i in samples]

    def sample_ops(self, n_samples):
        samples = self.prob().multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())

    def reset_ops(self):
        s_op = list(range(len(self.primitives())))
        self.last_samples = s_op
        self.s_op = s_op

    def forward(self, *args, **kwargs):
        self.sample_path()
        s_path_f = self.s_path_f
        primitives = self.primitives()
        if self.training: 
            self.swap_ops(s_path_f)
        if self.a_grad_enabled:
            p = self.arch_param_value('p')
            ctx_dict = {
                's_op': self.s_op,
                's_path_f': self.s_path_f,
                'w_path_f': self.w_path_f,
                'primitives': primitives,
            }
            m_out = BinGateFunction.apply(kwargs, p, ctx_dict, *args)
        else:
            outputs = [primitives[i](*args, **kwargs) for i in s_path_f]
            m_out = sum(outputs) if len(s_path_f) > 1 else outputs[0]
        self.last_samples = s_path_f
        return m_out

    def swap_ops(self, samples):
        prims = self.primitives()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in prims[i].parameters():
                if not p.grad_fn is None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in prims[i].parameters():
                if not p.grad_fn is None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []: return [None]
        return desc


class BinGateFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(ctx, kwargs, alpha, ctx_dict, *args):
        ctx.__dict__.update(ctx_dict)
        ctx.kwargs = kwargs
        ctx.param_shape = alpha.shape
        primitives = ctx.primitives
        s_path_f = ctx.s_path_f
        with torch.enable_grad():
            if len(s_path_f) == 1: m_out = primitives[s_path_f[0]](*args, **kwargs)
            else: m_out = sum(primitives[i](*args, **kwargs) for i in s_path_f)
        ctx.save_for_backward(*args, m_out)
        return m_out.data

    @staticmethod
    def backward(ctx, m_grad):
        args_f = ctx.saved_tensors[:-1]
        m_out = ctx.saved_tensors[-1]
        retain = True if len(args_f) > 1 else False
        grad_args = torch.autograd.grad(m_out, args_f, m_grad, only_inputs=True, retain_graph=retain)
        with torch.no_grad():
            a_grad = torch.zeros(ctx.param_shape)
            s_op = ctx.s_op
            w_path_f = ctx.w_path_f
            s_path_f = ctx.s_path_f
            kwargs = ctx.kwargs
            primitives = ctx.primitives
            for j, oj in enumerate(s_op):
                if oj in s_path_f:
                    op_out = m_out.data
                else:
                    op = primitives[oj]
                    op_out = op(*args_f, **kwargs)
                g_grad = torch.sum(m_grad * op_out)
                for i, oi in enumerate(s_op):
                    kron = 1 if i==j else 0
                    a_grad[oi] = a_grad[oi] + g_grad * w_path_f[j] * (kron - w_path_f[i])
        return (None, a_grad, None) + grad_args


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


class GumbelSumMixedOp(MixedOp):
    """ Mixed operation as weighted sum """
    def __init__(self, primitives, arch_param_map=None):
        super().__init__(primitives, arch_param_map)
        self.temp = 1e5

    def set_temperature(self, temp):
        self.temp = temp

    def prob(self):
        p = self.alpha()
        eps = 1e-7
        uniforms = torch.rand(p.shape, device=p.device).clamp(eps, 1-eps)
        gumbels = -((-(uniforms.log())).log())
        scores = (p + gumbels) / self.temp
        return F.softmax(scores, dim=-1)

    def forward(self, *args, **kwargs):
        outputs = [op(*args, **kwargs) for op in self.primitives()]
        w_path = self.prob().to(outputs[0].device)
        return sum(w * o for w, o in zip(w_path, outputs))

    def to_arch_desc(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1) # use alpha softmax
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []: return [None]
        return desc


class IndexMixedOp(MixedOp):
    """ Mixed operation controlled by external index """
    def __init__(self, primitives, arch_param_map=None):
        if arch_param_map is None:
            arch_param_map = {
                'prims': ArchParamCategorical(list(primitives.keys())),
            }
        super().__init__(primitives, arch_param_map)
        self.last_samples = []
        self.reset_ops()

    def alpha(self):
        alpha = torch.zeros(len(self.primitives()))
        alpha[self.arch_param('prims').index()] = 1.0
        return alpha

    def prob(self):
        return self.alpha()

    def forward(self, *args, **kwargs):
        prims = self.primitives()
        smp = self.arch_param('prims').index()
        if self.training: self.swap_ops([smp])
        self.last_samples = [smp]
        return prims[smp](*args, **kwargs)

    def reset_ops(self):
        s_op = list(range(len(self.primitives())))
        self.last_samples = s_op

    def swap_ops(self, samples):
        prims = self.primitives()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in prims[i].parameters():
                if not p.grad_fn is None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in prims[i].parameters():
                if not p.grad_fn is None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, *args, **kwargs):
        return self.arch_param_value('prims')


register(WeightedSumMixedOp, 'WeightedSum')
register(BinGateMixedOp, 'BinGate')
register(BinGateUniformMixedOp, 'BinGateUniform')
register(GumbelSumMixedOp, 'GumbelSum')
register(IndexMixedOp, 'Index')
