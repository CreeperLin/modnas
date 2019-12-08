import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..arch_space import genotypes as gt
from .ops import DropPath_
from .param_space import ArchParamSpace
from .nas_modules import NASModule

class NASController(nn.Module):
    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.net = net

    def forward(self, x):
        if len(self.device_ids) <= 1:
            return self.net(x)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)

        # replicate modules
        self.net.to(device=self.device_ids[0])
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(xs),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def loss_logits(self, X, y, aux_weight=0):
        ret = self.forward(X)
        if isinstance(ret, tuple):
            logits, aux_logits = ret
            aux_loss = aux_weight * self.criterion(aux_logits, y)
        else:
            logits = ret
            aux_loss = 0
        return self.criterion(logits, y) + aux_loss, logits
    
    def loss(self, X, y, aux_weight=0):
        loss, _ = self.loss_logits(X, y, aux_weight)
        return loss
    
    def logits(self, X, aux_weight=0):
        ret = self.forward(X)
        if isinstance(ret, tuple):
            logits, aux_logits = ret
            logits += aux_weight * aux_logits
        else:
            logits = ret
        return logits

    def print_arch_params(self, logger, max_num=3):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        alphas = tuple(F.softmax(a.detach(), dim=-1).cpu().numpy() for a in self.alphas())
        max_num = min(len(alphas)//2, max_num)
        logger.info("ALPHA: {}\n{}".format(
            len(alphas), '\n'.join([str(a) for a in (alphas[:max_num]+('...',)+alphas[-max_num:])])))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def dags(self):
        if not hasattr(self.net,'dags'): return []
        return self.net.dags()

    def to_genotype(self, *args, **kwargs):
        try:
            gene_dag = self.net.to_genotype(*args, **kwargs)
            return gt.Genotype(dag=gene_dag, ops=None)
        except AttributeError:
            return self.to_genotype_ops(*args, **kwargs)
    
    def to_genotype_ops(self, *args, **kwargs):
        gene_ops = NASModule.to_genotype_all(*args, **kwargs)
        return gt.Genotype(dag=None, ops=gene_ops)
    
    def build_from_genotype(self, gene, *args, **kwargs):
        try:
            self.net.build_from_genotype(gene, *args, **kwargs)
        except AttributeError:
            NASModule.build_from_genotype_all(gene, *args, **kwargs)

    def weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield p

    def named_weights(self, check_grad=False):
        for n, p in self.net.named_parameters(recurse=True):
            if check_grad and not p.requires_grad:
                continue
            yield n, p

    def alphas(self):
        return ArchParamSpace.continuous_params()

    def mixed_ops(self):
        return NASModule.modules()
    
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

    def init_model(self, config):
        init_type = config.type
        init_div_groups = config.conv_div_groups
        if init_type == 'none': return
        a = 0.
        gain = math.sqrt(2. / (1 + a**2))
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan = m.kernel_size[0] * m.kernel_size[1]
                if init_div_groups:
                    fan /= m.groups
                if init_type == 'he_normal_fout':
                    fan *= m.out_channels
                    stdv = gain / math.sqrt(fan)
                    nn.init.normal_(m.weight, 0, stdv)
                elif init_type == 'he_normal_fin':
                    fan *= m.in_channels
                    stdv = gain / math.sqrt(fan)
                    nn.init.normal_(m.weight, 0, stdv)
                elif init_type == 'he_uniform_fout':
                    fan *= m.out_channels
                    b = math.sqrt(3.) * gain / math.sqrt(fan)
                    nn.init.uniform_(m.weight, -b, b)
                elif init_type == 'he_uniform_fin':
                    fan *= m.in_channels
                    b = math.sqrt(3.) * gain / math.sqrt(fan)
                    nn.init.uniform_(m.weight, -b, b)
                else:
                    raise NotImplementedError
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if not m.weight is None: nn.init.ones_(m.weight)
                if not m.bias is None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight, -stdv, stdv)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
