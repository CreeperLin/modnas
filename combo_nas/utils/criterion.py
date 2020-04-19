# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('criterion')
from ..arch_space import build as build_arch_space
from ..arch_space.constructor import Slot, convert_from_predefined_net

def label_smoothing(y_pred, y_true, eta):
    n_classes = y_pred.size(1)
    # convert to one-hot
    y_true = torch.unsqueeze(y_true, 1)
    soft_y_true = torch.zeros_like(y_pred)
    soft_y_true.scatter_(1, y_true, 1)
    # label smoothing
    soft_y_true = soft_y_true * (1 - eta) + eta / n_classes * 1
    return soft_y_true


def cross_entropy_soft_target(y_pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(y_pred, dim=-1), 1))


@register_as('LabelSmoothing')
class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta

    def forward(self, y_pred, y_true):
        soft_y_true = label_smoothing(y_pred, y_true, self.eta)
        return cross_entropy_soft_target(y_pred, soft_y_true)


@register_as('Auxiliary')
class Auxiliary():
    def __init__(self, aux_ratio=0.4, loss_type='ce', forward_func='forward_aux'):
        super().__init__()
        self.aux_ratio = aux_ratio
        self.fwd_func = forward_func
        if loss_type == 'ce':
            self.loss_func = F.cross_entropy
        else:
            raise ValueError('unsupported loss type: {}'.format(loss_type))

    def __call__(self, loss, estim, X, y_pred, y_true):
        aux_logits = estim.model.call(self.fwd_func, X)
        if aux_logits is None:
            return loss
        aux_loss = self.loss_func(aux_logits, y_true).to(device=loss.device)
        return loss + self.aux_ratio * aux_loss


@register_as('KnowledgeDistill')
class KnowledgeDistill():
    def __init__(self, kd_model_path, kd_model_type, kd_model_args={},
                 kd_model=None, kd_ratio=0.5, loss_scale=1., loss_type='ce'):
        super().__init__()
        if kd_model is None:
            kd_model = self.load_model(kd_model_path, kd_model_type, kd_model_args)
        self.kd_model = kd_model
        self.kd_ratio = kd_ratio
        self.loss_scale = loss_scale
        if loss_type == 'ce':
            self.loss_func = lambda y_pred, target: cross_entropy_soft_target(y_pred, F.softmax(target, dim=-1))
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise ValueError('unsupported loss_type: {}'.format(loss_type))

    def load_model(self, model_path, model_type, model_args):
        model = build_arch_space(model_type, **model_args)
        convert_fn = model.get_predefined_augment_converter()
        model = convert_from_predefined_net(model, convert_fn, gen=Slot.gen_slots_model(model))
        model.train()
        if not model_path is None:
            state_dict = torch.load(model_path)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
        return model

    def __call__(self, loss, estim, X, y_pred, y_true):
        with torch.no_grad():
            self.kd_model.to(device=X.device)
            soft_logits = self.kd_model(X)
        kd_loss = self.loss_func(y_pred, soft_logits).to(device=loss.device)
        loss = self.loss_scale * ((1 - self.kd_ratio) * loss + self.kd_ratio * kd_loss)
        return loss


@register_as('AddMetrics')
class AddMetrics():
    def __init__(self, target_val, metrics, lamd=0.01,):
        super().__init__()
        self.lamd = lamd
        self.target_val = float(target_val)
        self.metrics = metrics

    def __call__(self, loss, estim, X, y_pred, y_true):
        mt = estim.compute_metrics(name=self.metrics).to(device=loss.device)
        return loss + self.lamd * (mt / self.target_val - 1.)


@register_as('MultMetrics')
class MultMetrics():
    def __init__(self, target_val, metrics, alpha=1., beta=0.6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)
        self.metrics = metrics

    def __call__(self, loss, estim, X, y_pred, y_true):
        mt = estim.compute_metrics(name=self.metrics).to(device=loss.device)
        return self.alpha * loss * (mt / self.target_val) ** self.beta


@register_as('MultLogMetrics')
class MultLogMetrics():
    def __init__(self, target_val, metrics, alpha=1., beta=0.6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.target_val = float(target_val)
        self.metrics = metrics

    def __call__(self, loss, estim, X, y_pred, y_true):
        mt = estim.compute_metrics(name=self.metrics).to(device=loss.device)
        return self.alpha * loss * (torch.log(mt) / math.log(self.target_val)) ** self.beta


register(nn.CrossEntropyLoss, 'CE')
