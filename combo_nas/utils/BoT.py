# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def label_smoothing(y_pred, y_true, eta):
    n_classes = y_pred.size(1)
    # convert to one-hot
    y_true = torch.unsqueeze(y_true, 1)
    soft_y_true = torch.zeros_like(y_pred)
    soft_y_true.scatter_(1, y_true, 1)
    # label smoothing
    soft_y_true = soft_y_true * (1 - eta) + eta / n_classes * 1
    return soft_y_true


class CrossEntropyLossLS(nn.Module):
    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta

    def forward(self, y_pred, y_true):
        soft_y_true = label_smoothing(y_pred, y_true, self.eta)
        return torch.mean(torch.sum(-soft_y_true * F.log_softmax(y_pred, dim=-1), 1))


class CrossEntropyLossMixup(nn.Module):
    def __init__(self, lam=1, eta=0.1):
        super().__init__()
        self.eta = eta
        self.lam = lam

    def forward(self, y_pred, y_true):
        onehot_target = label_smoothing(y_pred, y_true, self.eta)
        flipped_target = onehot_target[::-1]  # flip over batch dimensions
        mixed_y_true = self.lam * onehot_target + (1 - self.lam) * flipped_target
        return torch.mean(torch.sum(-mixed_y_true * F.log_softmax(y_pred, dim=-1), 1))