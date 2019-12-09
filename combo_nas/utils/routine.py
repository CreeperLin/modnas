# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .. import utils
from .visualize import plot
from .profiling import tprof
from ..arch_space import genotypes as gt
from ..core.nas_modules import ArchModuleSpace
from ..estimator import build_estimator

def search(config, chkpt_path, expman, train_loader, valid_loader, model, arch_optim, writer, logger, device):
    logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logger.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim = build_estimator(estim_type, estim_conf, expman, train_loader, valid_loader, model, writer, logger, device)
        estim.load(chkpt_path)
        ret = estim.search(arch_optim)
        estim.save(-1)
    best_top1, best_genotype, genotypes = ret
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    gt.to_file(best_genotype, expman.join('output', 'best.gt'))
    return best_top1, best_genotype, genotypes


def augment(config, chkpt_path, expman, train_loader, valid_loader, model, writer, logger, device):
    logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logger.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim = build_estimator(estim_type, estim_conf, expman, train_loader, valid_loader, model, writer, logger, device)
        estim.load(chkpt_path)
        ret = estim.train()
        estim.save_checkpoint(-1)
    best_top1 = ret
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1
