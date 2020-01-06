# -*- coding: utf-8 -*-
from .. import utils
from ..arch_space import genotypes as gt
from ..estimator import build_estimator

def search(config, chkpt_path, expman, train_loader, valid_loader, model, arch_optim, writer, logger, device):
    if not model is None:
        logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logger.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim = build_estimator(estim_type, estim_conf, expman, train_loader, valid_loader, model, writer, logger, device)
        estim.load(chkpt_path)
        ret = estim.search(arch_optim)
    best_top1, best_genotype, genotypes = ret
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    gt.to_file(best_genotype, expman.join('output', 'best.gt'))
    return best_top1, best_genotype, genotypes


def augment(config, chkpt_path, expman, train_loader, valid_loader, model, writer, logger, device):
    if not model is None:
        logger.info("Model params count: {:.3f} M, size: {:.3f} MB".format(utils.param_count(model), utils.param_size(model)))
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logger.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim = build_estimator(estim_type, estim_conf, expman, train_loader, valid_loader, model, writer, logger, device)
        estim.load(chkpt_path)
        ret = estim.train()
    best_top1 = ret
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1


def hptune(config, chkpt_path, expman, optim, writer, logger, device, measure_fn):
    del chkpt_path
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logger.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim = build_estimator(estim_type, estim_conf, expman, writer, logger, device, measure_fn)
        ret = estim.search(optim)
    best_iter, best_score, best_hparams = ret
    logger.info('hptune: finished: best iter: {} score: {} config: {}'.format(best_iter, best_score, best_hparams))
    return ret
