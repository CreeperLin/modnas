# -*- coding: utf-8 -*-
""" Genotypes
"""
import logging
import os
from collections import namedtuple

Genotype = namedtuple('Genotype', 'dag ops')

def to_file(gene, path):
    g_str = str(gene)
    try:
        with open(path, 'w', encoding='UTF-8') as f:
            f.write(g_str)
    except Exception as e:
        logging.info('failed saving genotype: {}'.format(e))


def from_file(path):
    if not os.path.exists(path):
        logging.debug("genotype file not found: {}".format(path))
        return Genotype(dag=None, ops=None)
    with open(path, 'r', encoding='UTF-8') as f:
        g_str = f.read()
    return from_str(g_str)


def from_str(g_str):
    """ generate genotype from string """
    genotype = eval(g_str)
    return genotype


def get_genotype(config, ovr_genotype):
    if isinstance(ovr_genotype, Genotype):
        return ovr_genotype
    elif isinstance(ovr_genotype, str):
        if ovr_genotype.startswith('Genotype'):
            return from_str(ovr_genotype)
        else: return from_file(ovr_genotype)
    elif config.gt_str == '':
        return from_file(config.gt_path)
    else:
        return from_str(config.gt_str)
    return None
