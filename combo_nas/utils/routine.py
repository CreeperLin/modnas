# -*- coding: utf-8 -*-
import logging
from ..arch_space import genotypes as gt
from ..estimator import build as build_estimator

def search(config, chkpt_path, optim, estim_kwargs):
    expman = estim_kwargs['expman']
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_build_kwargs = estim_conf.get('args', {})
        estim_build_kwargs.update(estim_kwargs)
        estim_build_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_build_kwargs)
        estim.load(chkpt_path)
        ret = estim.search(optim)
    logging.info('Final results:\n{}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in ret.items()])))
    if 'best_gt' in ret:
        gt.to_file(ret['best_gt'], expman.join('output', 'best.gt'))
    return ret


def augment(config, chkpt_path, estim_kwargs):
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_build_kwargs = estim_conf.get('args', {})
        estim_build_kwargs.update(estim_kwargs)
        estim_build_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_build_kwargs)
        estim.load(chkpt_path)
        ret = estim.train()
    logging.info('Final results:\n{}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in ret.items()])))
    return ret


def hptune(config, chkpt_path, optim, estim_kwargs):
    for estim_name, estim_conf in config.items():
        estim_type = estim_conf.type
        logging.info('running estimator: {} type: {}'.format(estim_name, estim_type))
        estim_build_kwargs = estim_conf.get('args', {})
        estim_build_kwargs.update(estim_kwargs)
        estim_build_kwargs['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_build_kwargs)
        estim.load(chkpt_path)
        ret = estim.search(optim)
    logging.info('Final results:\n{}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in ret.items()])))
    return ret
