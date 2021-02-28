"""Mixed operators."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.params import Categorical
from modnas.registry.params import build
from modnas.registry.arch_space import register
from modnas.utils.logging import get_logger


logger = get_logger('arch_space')


class MixedOp(nn.Module):
    """Base Mixed operator class."""

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
                'default': build('TorchTensor', len(self._ops)),
            }
        self.arch_param_map = arch_param_map
        logger.debug('mixed op: {} p: {}'.format(type(self), arch_param_map))

    def primitives(self):
        """Return list of primitive operators."""
        return list(self._ops.values())

    def primitive_names(self):
        """Return list of primitive operator names."""
        return list(self._ops.keys())

    def named_primitives(self):
        """Return an iterator over named primitive operators."""
        for n, prim in self._ops.items():
            yield n, prim

    def alpha(self):
        """Return architecture parameter value."""
        return self.arch_param_value()

    def prob(self):
        """Return primitive probabilities."""
        return F.softmax(self.alpha(), dim=-1)

    def arch_param(self, name='default'):
        """Return architecture parameter by name."""
        return self.arch_param_map.get(name)

    def arch_param_value(self, name='default'):
        """Return architecture parameter value by name."""
        return self.arch_param_map.get(name).value()

    def to_arch_desc(self, *args, **kwargs):
        """Return archdesc from mixed operator."""
        raise NotImplementedError

    @staticmethod
    def gen(model):
        """Return an iterator over all MixedOp in a model."""
        for m in model.modules():
            if isinstance(m, MixedOp):
                yield m


@register
class SoftmaxSumMixedOp(MixedOp):
    """Mixed operator using softmax weighted sum."""

    def __init__(self, primitives, arch_param_map=None):
        super().__init__(primitives, arch_param_map)

    def forward(self, *args, **kwargs):
        """Compute MixedOp output."""
        outputs = [op(*args, **kwargs) for op in self.primitives()]
        w_path = F.softmax(self.alpha().to(device=outputs[0].device), dim=-1)
        return sum(w * o for w, o in zip(w_path, outputs))

    def to_arch_desc(self, k=1):
        """Return archdesc from mixed operator."""
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []:
            return [None]
        return desc


@register
class BinaryGateMixedOp(MixedOp):
    """Mixed operator controlled by BinaryGate."""

    def __init__(self, primitives, arch_param_map=None, n_samples=1):
        super().__init__(primitives, arch_param_map)
        self.n_samples = n_samples
        self.s_path_f = None
        self.last_samples = []
        self.s_op = []
        self.a_grad_enabled = False
        self.reset_ops()

    def arch_param_grad(self, enabled):
        """Set if enable architecture parameter grad."""
        self.a_grad_enabled = enabled

    def sample_path(self):
        """Sample primitives in forward pass."""
        p = self.alpha()
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = self.w_path_f.multinomial(1 if self.a_grad_enabled else self.n_samples)
        self.s_path_f = [s_op[i] for i in samples]

    def sample_ops(self, n_samples):
        """Sample activated primitives."""
        samples = self.prob().multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())

    def reset_ops(self):
        """Reset activated primitives."""
        s_op = list(range(len(self.primitives())))
        self.last_samples = s_op
        self.s_op = s_op

    def forward(self, *args, **kwargs):
        """Compute MixedOp output."""
        self.sample_path()
        s_path_f = self.s_path_f
        primitives = self.primitives()
        if self.training:
            self.swap_ops(s_path_f)
        if self.a_grad_enabled:
            p = self.alpha()
            ctx_dict = {
                's_op': self.s_op,
                's_path_f': self.s_path_f,
                'w_path_f': self.w_path_f,
                'primitives': primitives,
            }
            m_out = BinaryGateFunction.apply(kwargs, p, ctx_dict, *args)
        else:
            outputs = [primitives[i](*args, **kwargs) for i in s_path_f]
            m_out = sum(outputs) if len(s_path_f) > 1 else outputs[0]
        self.last_samples = s_path_f
        return m_out

    def swap_ops(self, samples):
        """Remove unused primitives from computation graph."""
        prims = self.primitives()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in prims[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in prims[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, k=1):
        """Return archdesc from mixed operator."""
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []:
            return [None]
        return desc


class BinaryGateFunction(torch.autograd.function.Function):
    """BinaryGate gradient approximation function."""

    @staticmethod
    def forward(ctx, kwargs, alpha, ctx_dict, *args):
        """Return forward outputs."""
        ctx.__dict__.update(ctx_dict)
        ctx.kwargs = kwargs
        ctx.param_shape = alpha.shape
        primitives = ctx.primitives
        s_path_f = ctx.s_path_f
        with torch.enable_grad():
            if len(s_path_f) == 1:
                m_out = primitives[s_path_f[0]](*args, **kwargs)
            else:
                m_out = sum(primitives[i](*args, **kwargs) for i in s_path_f)
        ctx.save_for_backward(*args, m_out)
        return m_out.data

    @staticmethod
    def backward(ctx, m_grad):
        """Return backward outputs."""
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
                    kron = 1 if i == j else 0
                    a_grad[oi] = a_grad[oi] + g_grad * w_path_f[j] * (kron - w_path_f[i])
        return (None, a_grad, None) + grad_args


@register
class BinaryGateUniformMixedOp(BinaryGateMixedOp):
    """Mixed operator controlled by BinaryGate, which primitives sampled uniformly."""

    def sample_path(self):
        """Sample primitives in forward pass."""
        p = self.alpha()
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        # sample uniformly
        samples = F.softmax(torch.ones(len(s_op)), dim=-1).multinomial(self.n_samples)
        s_path_f = [s_op[i] for i in samples]
        self.s_path_f = s_path_f

    def sample_ops(self, n_samples):
        """Sample activated primitives."""
        p = self.alpha()
        # sample uniformly
        samples = F.softmax(torch.ones(p.shape), dim=-1).multinomial(n_samples).detach()
        # sample according to p
        # samples = F.softmax(p, dim=-1).multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())


@register
class GumbelSumMixedOp(MixedOp):
    """Mixed operator using gumbel softmax sum."""

    def __init__(self, primitives, arch_param_map=None):
        super().__init__(primitives, arch_param_map)
        self.temp = 1e5

    def set_temperature(self, temp):
        """Set annealing temperature."""
        self.temp = temp

    def prob(self):
        """Return primitive probabilities."""
        p = self.alpha()
        eps = 1e-7
        uniforms = torch.rand(p.shape, device=p.device).clamp(eps, 1 - eps)
        gumbels = -((-(uniforms.log())).log())
        scores = (p + gumbels) / self.temp
        return F.softmax(scores, dim=-1)

    def forward(self, *args, **kwargs):
        """Compute MixedOp output."""
        outputs = [op(*args, **kwargs) for op in self.primitives()]
        w_path = self.prob().to(outputs[0].device)
        return sum(w * o for w, o in zip(w_path, outputs))

    def to_arch_desc(self, k=1):
        """Return archdesc from mixed operator."""
        pname = self.primitive_names()
        w = F.softmax(self.alpha().detach(), dim=-1)  # use alpha softmax
        _, prim_idx = torch.topk(w, k)
        desc = [pname[i] for i in prim_idx]
        if desc == []:
            return [None]
        return desc


@register
class IndexMixedOp(MixedOp):
    """Mixed operator controlled by index."""

    def __init__(self, primitives, arch_param_map=None):
        if arch_param_map is None:
            arch_param_map = {
                'prims': Categorical(list(primitives.keys())),
            }
        super().__init__(primitives, arch_param_map)
        self.last_samples = list(range(len(self.primitives())))

    def alpha(self):
        """Return architecture parameter value."""
        alpha = torch.zeros(len(self.primitives()))
        alpha[self.arch_param('prims').index()] = 1.0
        return alpha

    def forward(self, *args, **kwargs):
        """Compute MixedOp output."""
        prims = self.primitives()
        smp = self.arch_param('prims').index()
        if self.training:
            self.swap_ops([smp])
        self.last_samples = [smp]
        return prims[smp](*args, **kwargs)

    def swap_ops(self, samples):
        """Remove unused primitives from computation graph."""
        prims = self.primitives()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in prims[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in prims[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, *args, **kwargs):
        """Return archdesc from mixed operator."""
        return self.arch_param_value('prims')
