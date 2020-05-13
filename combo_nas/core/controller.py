import traceback
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..arch_space import genotypes as gt
from ..arch_space.mixed_ops import MixedOp
from ..arch_space.constructor import Slot
from .param_space import ArchParamSpace
from ..utils.init import DefaultModelInitializer

class NASController(nn.Module):
    def __init__(self, net, device_ids=None):
        super().__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        if len(device_ids) > 0:
            net = net.to(device=device_ids[0])
        if len(device_ids) > 1:
            net = nn.parallel.DataParallel(net, device_ids=device_ids)
        self._net = net
        self.to_genotype_args = {}

    @property
    def net(self):
        if isinstance(self._net, nn.parallel.DataParallel):
            return self._net.module
        return self._net

    def call(self, func, *args, **kwargs):
        return getattr(self.net, func)(*args, **kwargs)

    def forward(self, x):
        return self._net(x)

    def logits(self, X):
        logits = self.forward(X)
        return logits

    def print_arch_params(self, logger, max_num=3):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        ap_cont = tuple(F.softmax(a.detach(), dim=-1).cpu().numpy() for a in self.arch_param_tensor())
        ap_disc = tuple(self.arch_param_categorical())
        max_num = min(len(ap_cont)//2, max_num)
        if len(ap_cont) != 0: logger.info("TENSOR: {}\n{}".format(
            len(ap_cont), '\n'.join([str(a) for a in (ap_cont[:max_num]+('...',)+ap_cont[-max_num:])])))
        max_num = min(len(ap_disc)//2, max_num)
        if len(ap_disc) != 0: logger.info("CATEGORICAL: {}\n{}".format(
            len(ap_disc), '\n'.join([str(a) for a in (ap_disc[:max_num]+('...',)+ap_disc[-max_num:])])))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def dags(self):
        if not hasattr(self.net,'dags'): return []
        return self.net.dags()

    def to_genotype(self, *args, **kwargs):
        if hasattr(self.net, 'to_genotype'):
            gene_dag = self.net.to_genotype(*args, **kwargs, **self.to_genotype_args)
            return gt.Genotype(dag=gene_dag, ops=None)
        else:
            return self.to_genotype_ops(*args, **kwargs)

    def to_genotype_fallback(self, *args, **kwargs):
        return {k: v.value() for k, v in ArchParamSpace.named_params()}

    def to_genotype_ops(self, *args, **kwargs):
        gene_ops = [m.to_genotype(*args, **kwargs, **self.to_genotype_args) for m in self.mixed_ops()]
        return gt.Genotype(dag=None, ops=gene_ops)

    def to_genotype_slots(self, *args, **kwargs):
        gene_ops = Slot.to_genotype_all(*args, **kwargs, **self.to_genotype_args)
        return gt.Genotype(dag=None, ops=gene_ops)

    def weights(self, check_grad=False):
        for p in self.net.parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield p

    def named_weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield n, p

    def arch_param_tensor(self):
        return ArchParamSpace.tensor_values()

    def arch_param_categorical(self):
        return ArchParamSpace.categorical_values()

    def alphas(self):
        return self.arch_param_tensor()

    def mixed_ops(self):
        for m in self.modules():
            if isinstance(m, MixedOp):
                yield m

    def save(self, path):
        try:
            torch.save(self.net.state_dict(), path)
            logging.info('Saved model checkpoint to: {}'.format(path))
        except:
            logging.error('Failed saving model: {}'.format(traceback.format_exc()))

    def load(self, path):
        sd = torch.load(path)
        self.net.load_state_dict(sd)
        logging.info('Loaded model from: {}'.format(path))

    def init_model(self, *args, net_init_fn=None, net_init_kwargs=None, **kwargs):
        init = DefaultModelInitializer(*args, **kwargs)
        init(self.net)
        if net_init_fn and hasattr(self.net, net_init_fn):
            self.net.init_model(**(net_init_kwargs or {}))
