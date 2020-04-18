import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.param_space import ArchParamCategorical, ArchParamTensor
from ..utils.registration import get_registry_utils

registry, register, get_builder, build, register_as = get_registry_utils('mixed_ops')

class MixedOp(nn.Module):
    def __init__(self, primitives, arch_param_map):
        super().__init__()
        if isinstance(primitives, dict):
            self._ops_name = list(primitives.keys())
            self._ops = nn.ModuleList(primitives.values())
        elif isinstance(primitives, (tuple, list)):
            self._ops_name = [p[0] for p in primitives]
            self._ops = nn.ModuleList([p[1] for p in primitives])
        else:
            raise ValueError('unsupported primitives type')
        if arch_param_map is None:
            arch_param_map = {
                'p': ArchParamTensor(len(self._ops_name)),
            }
        self.arch_param_map = arch_param_map
        for ap in arch_param_map.values():
            ap.add_module(self)
        self.w_max = 0
        logging.debug('mixed op: {} p: {}'.format(type(self), arch_param_map))

    def primitives(self):
        return self._ops

    def primitive_names(self):
        return self._ops_name

    def named_primitives(self):
        for n, prim in zip(self._ops_name, self._ops):
            yield n, prim

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
    def __init__(self, primitives, arch_param_map=None):
        super().__init__(primitives, arch_param_map)

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        w_path = F.softmax(self.alpha().to(device=x.device), dim=-1)
        return sum(w * op(x) for w, op in zip(w_path, self.primitives()))

    def to_genotype(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, k)
        gene = [pname[i] for i in prim_idx]
        if gene == []: return [None]
        self.w_max = w_max
        return gene


class DARTSMixedOp(WeightedSumMixedOp):
    def to_genotype(self, k=1):
        pname = self.primitive_names()
        assert pname[-1] == 'NIL'
        w = F.softmax(self.alpha().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w[:-1], k)
        gene = [pname[i] for i in prim_idx]
        if gene == []: return [None]
        self.w_max = w_max
        return gene


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

    def forward(self, x):
        self.sample_path()
        x = x[0] if isinstance(x, list) else x
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
            m_out = BinGateFunction.apply(x, p, ctx_dict)
        else:
            if len(s_path_f) == 1: m_out = primitives[s_path_f[0]](x)
            else: m_out = sum(primitives[i](x) for i in s_path_f)
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

    def to_genotype(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        w_max, prim_idx = torch.topk(w, k)
        gene = [pname[i] for i in prim_idx]
        if gene == []: return [None]
        self.w_max = w_max
        return gene


class BinGateFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(ctx, x, alpha, ctx_dict):
        ctx.__dict__.update(ctx_dict)
        ctx.param_shape = alpha.shape
        primitives = ctx.primitives
        s_path_f = ctx.s_path_f
        with torch.enable_grad():
            if len(s_path_f) == 1: m_out = primitives[s_path_f[0]](x)
            else: m_out = sum(primitives[i](x) for i in s_path_f)
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
            primitives = ctx.primitives
            for j, oj in enumerate(s_op):
                if oj in s_path_f:
                    op_out = m_out.data
                else:
                    op = primitives[oj]
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

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        w_path = self.prob().to(x.device)
        return sum(w * op(x) for w, op in zip(w_path, self.primitives()))

    def to_genotype(self, k=1):
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1) # use alpha softmax
        w_max, prim_idx = torch.topk(w, k)
        gene = [pname[i] for i in prim_idx]
        if gene == []: return [None]
        self.w_max = w_max
        return gene


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

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        smp = self.arch_param('prims').index()
        if self.training: self.swap_ops([smp])
        self.last_samples = [smp]
        return self._ops[smp](x)

    def reset_ops(self):
        s_op = list(range(len(self._ops)))
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

    def to_genotype(self, *args, **kwargs):
        return self.arch_param_value('prims')


register(WeightedSumMixedOp, 'WeightedSum')
register(DARTSMixedOp, 'DARTS')
register(BinGateMixedOp, 'BinGate')
register(BinGateUniformMixedOp, 'BinGateUniform')
register(GumbelSumMixedOp, 'GumbelSum')
register(IndexMixedOp, 'Index')
