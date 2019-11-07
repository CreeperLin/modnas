# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..arch_space import genotypes as gt
from .ops import DropPath_, configure_ops
from ..utils.profiling import tprof
from ..utils import param_count, get_current_device, get_net_crit
import traceback

class NASModule(nn.Module):
    _init_ratio = 1e-3
    _modules = []
    _params = []
    _module_id = -1
    _module_state_dict = {}
    _param_id = -1
    _params_map = {}
    _dev_list = [get_current_device()]

    def __init__(self, params_shape, pid=None):
        super().__init__()
        self.id = self.get_new_id()
        if pid==-1:
            self.pid = NASModule._param_id
        elif pid is None:
            self.pid = NASModule.add_param(params_shape)
        elif not isinstance(pid, int) or pid > NASModule._param_id:
            raise ValueError('invalid pid: {}'.format(pid))
        elif NASModule._params[pid].shape != params_shape:
            raise ValueError('size mismatch for given pid: {} {}'.format(pid, params_shape))
        else:
            self.pid = pid
        NASModule.add_module(self, self.id, self.pid)
        # logging.debug('reg nas module: {} mid: {} pid: {} {}'.format(self.__class__.__name__, self.id, self.pid, params_shape))
    
    @staticmethod
    def reset():
        NASModule._modules = []
        NASModule._params = []
        NASModule._module_id = -1
        NASModule._module_state_dict = {}
        NASModule._param_id = -1
        NASModule._params_map = {}
        NASModule._dev_list = [get_current_device()]

    @property
    def arch_param(self):
        return NASModule._params[self.pid]

    @staticmethod
    def nasmod_state_dict():
        return {
            # '_modules': NASModule._modules,
            '_params': NASModule._params,
            # '_module_id': NASModule._module_id,
            # '_module_state_dict': NASModule._module_state_dict,
            # '_param_id': NASModule._param_id,
            # '_params_map': NASModule._params_map
        }
    
    @staticmethod
    def nasmod_load_state_dict(sd):
        assert len(sd['_params']) == NASModule._param_id + 1
        for p, sp in zip(NASModule._params, sd['_params']):
            p.data.copy_(sp)

    @staticmethod
    def set_device(dev_list):
        dev_list = dev_list if len(dev_list)>0 else [None]
        NASModule._dev_list = [NASModule.get_dev_id(d) for d in dev_list]
    
    @staticmethod
    def get_device():
        return NASModule._dev_list    
    
    @staticmethod
    def get_dev_id(index):
        return 'cpu' if index is None else 'cuda:{}'.format(index)

    @staticmethod
    def get_new_id():
        NASModule._module_id += 1
        return NASModule._module_id
    
    @staticmethod
    def add_param(params_shape):
        NASModule._param_id += 1
        param = nn.Parameter(NASModule._init_ratio * torch.randn(params_shape).to(device=NASModule._dev_list[0]))
        NASModule._params.append(param)
        NASModule._params_map[NASModule._param_id] = []
        return NASModule._param_id

    @staticmethod
    def add_module(module, mid, pid):
        NASModule._modules.append(module)
        if pid >= 0: NASModule._params_map[pid].append(mid)
    
    @staticmethod
    def param_forward_all(params=None):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            for mid in pmap[pid]:
                mmap[mid].param_forward(NASModule._params[pid])
    
    @staticmethod
    def modules():
        for m in NASModule._modules:
            yield m
    
    @staticmethod
    def param_modules():
        mmap = NASModule._modules
        pmap = NASModule._params_map
        plist = NASModule._params
        for pid in pmap:
            mlist = pmap[pid]
            for mid in mlist:
                yield (plist[pid], mmap[pid])
    
    @staticmethod
    def params_grad(loss):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = mmap[mlist[0]].param_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].param_grad(loss)
            yield p_grad
    
    @staticmethod
    def param_backward(loss):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = mmap[mlist[0]].param_grad(loss)
            for i in range(1,len(mlist)):
                p_grad += mmap[mlist[i]].param_grad(loss)
            NASModule._params[pid].grad = p_grad

    @staticmethod
    def backward_all(loss):
        m_out_all = []
        m_out_len = []
        for dev_id in NASModule.get_device():
            m_out = [m.get_state('m_out'+dev_id) for m in NASModule.modules()]
            m_out_all.extend(m_out)
            m_out_len.append(len(m_out))
        m_grad = torch.autograd.grad(loss, m_out_all)
        for i, dev_id in enumerate(NASModule.get_device()):
            NASModule.param_backward_from_grad(m_grad[sum(m_out_len[:i]) : sum(m_out_len[:i+1])], dev_id)

    @staticmethod
    def param_backward_from_grad(m_grad, dev_id):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            mlist = pmap[pid]
            p_grad = 0
            for i in range(0,len(mlist)):
                p_grad = mmap[mlist[i]].param_grad_dev(m_grad[mlist[i]], dev_id) + p_grad
            if NASModule._params[pid].grad is None:
                NASModule._params[pid].grad = p_grad
            else:
                NASModule._params[pid].grad += p_grad
    
    @staticmethod
    def params():
        for p in NASModule._params:
            yield p
    
    @staticmethod
    def module_apply(func, **kwargs):
        return [func(m, **kwargs) for m in NASModule._modules]
    
    @staticmethod
    def module_call(func, **kwargs):
        return [getattr(m, func)(**kwargs) for m in NASModule._modules]
    
    @staticmethod
    def param_call(func, **kwargs):
        return [getattr(p, func)(**kwargs) for p in NASModule._params]
    
    @staticmethod
    def param_module_call(func, **kwargs):
        mmap = NASModule._modules
        pmap = NASModule._params_map
        for pid in pmap:
            for mid in pmap[pid]:
                getattr(mmap[mid], func)(NASModule._params[pid], **kwargs)

    def nas_state_dict(self):
        if not self.id in NASModule._module_state_dict:
            NASModule._module_state_dict[self.id] = {}
        return NASModule._module_state_dict[self.id]
    
    def get_state(self, name, detach=False):
        sd = self.nas_state_dict()
        if not name in sd: return
        ret = sd[name]
        ret = ret.detach() if detach else ret
        return ret
    
    def set_state(self, name, val, detach=False):
        val = val.detach() if detach else val
        self.nas_state_dict()[name] = val
    
    def del_state(self, name):
        if not name in self.nas_state_dict(): return
        del self.nas_state_dict()[name]
    
    @staticmethod
    def build_from_genotype_all(gene, *args, **kwargs):
        if gene.ops is None: return
        assert len(NASModule._modules) == len(gene.ops)
        for m, g in zip(NASModule._modules, gene.ops):
            m.build_from_genotype(g, *args, **kwargs)
    
    @staticmethod
    def to_genotype_all(*args, **kwargs):
        gene = []
        for m in NASModule._modules:
            _, g_module = m.to_genotype(*args, **kwargs)
            gene.append(g_module)
        return gene

    def build_from_genotype(self, gene, *args, **kwargs):
        raise NotImplementedError
    
    def to_genotype(self, *args, **kwargs):
        raise NotImplementedError


class NASController(nn.Module):
    def __init__(self, net, criterion, device_ids=None):
        super().__init__()
        self.criterion = criterion
        self.augment = len(tuple(self.alphas())) == 0
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.net = net

    def forward(self, x):
        if not self.augment: NASModule.param_forward_all()
        
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

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        alphas = tuple(F.softmax(a.detach(), dim=-1).cpu().numpy() for a in self.alphas())
        logger.info("ALPHA: {}\n{}".format(len(alphas), '\n'.join([str(a) for a in alphas])))

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
            # traceback.print_exc()
            gene_ops = NASModule.to_genotype_all(*args, **kwargs)
            return gt.Genotype(dag=None, ops=gene_ops)
    
    def build_from_genotype(self, gene, *args, **kwargs):
        try:
            self.net.build_from_genotype(gene, *args, **kwargs)
        except AttributeError:
            # traceback.print_exc()
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
        return NASModule.params()

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def alpha_grad(self, loss):
        return NASModule.params_grad()

    def alpha_backward(self, loss):
        NASModule.param_backward(loss)
    
    def mixed_ops(self):
        return NASModule.modules()
    
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

def build_nas_controller(net, crit, device, dev_list, verbose=False):
    NASModule.set_device(dev_list)
    model = NASController(net, crit, dev_list).to(device=device)
    if verbose: logging.debug(model)
    return model