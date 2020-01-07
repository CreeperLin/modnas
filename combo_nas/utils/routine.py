# -*- coding: utf-8 -*-
import logging
from .. import utils
from ..arch_space import genotypes as gt
from ..estimator import build_estimator

def search(config, chkpt_path, arch_optim, estim_kwargs):
    estimators = config.estimator
    expman = estim_kwargs['expman']
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_kwargs)
        estim.load(chkpt_path)
        ret = estim.search(arch_optim)
    best_top1, best_genotype, genotypes = ret
    logging.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logging.info("Best Genotype = {}".format(best_genotype))
    gt.to_file(best_genotype, expman.join('output', 'best.gt'))
    return best_top1, best_genotype, genotypes


def augment(config, chkpt_path, estim_kwargs):
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_kwargs)
        estim.load(chkpt_path)
        ret = estim.train()
    best_top1 = ret
    logging.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1


def hptune(config, chkpt_path, optim, estim_kwargs):
    del chkpt_path
    estimators = config.estimator
    for estim_name, estim_conf in estimators.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_kwargs)
        ret = estim.search(optim)
    best_iter, best_score, best_hparams = ret
    logging.info('hptune: finished: best iter: {} score: {} config: {}'.format(best_iter, best_score, best_hparams))
    return ret
