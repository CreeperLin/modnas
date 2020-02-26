# -*- coding: utf-8 -*-
import logging
from ..arch_space import genotypes as gt
from ..estimator import build as build_estimator

def search(config, chkpt_path, optim, estim_kwargs):
    expman = estim_kwargs['expman']
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim_kwargs.update(estim_conf.get('args', {}))
        estim = build_estimator(estim_type, **estim_kwargs)
        estim.load(chkpt_path)
        ret = estim.search(optim)
    logging.info("Final best Prec@1 = {:.4%}".format(ret['best_top1']))
    logging.info("Best Genotype = {}".format(ret['best_gt']))
    gt.to_file(ret['best_gt'], expman.join('output', 'best.gt'))
    return ret


def augment(config, chkpt_path, estim_kwargs):
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim_kwargs.update(estim_conf.get('args', {}))
        estim = build_estimator(estim_type, **estim_kwargs)
        estim.load(chkpt_path)
        ret = estim.train()
    logging.info("Final best Prec@1 = {:.4%}".format(ret['best_top1']))
    return ret


def hptune(config, chkpt_path, optim, estim_kwargs):
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_kwargs['config'] = estim_conf
        estim_kwargs.update(estim_conf.get('args', {}))
        estim = build_estimator(estim_type, **estim_kwargs)
        estim.load(chkpt_path)
        ret = estim.search(optim)
    logging.info('hptune: finished: best iter: {} score: {} config: {}'.format(
        ret['best_iter'], ret['best_score'], ret['best_hparams']))
    return ret
