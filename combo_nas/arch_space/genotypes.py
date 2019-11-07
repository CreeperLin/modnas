# -*- coding: utf-8 -*-
""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import logging
import os
from collections import namedtuple
from ..core import ops

Genotype = namedtuple('Genotype', 'dag ops')

PRIMITIVES = []

def set_primitives(prim):
    global PRIMITIVES
    PRIMITIVES = prim
    logging.info('candidate ops: {}'.format(get_primitives()))

def get_primitives():
    return PRIMITIVES

def to_file(gene, path):
    g_str = str(gene)
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(g_str)

def from_file(path):
    if not os.path.exists(path):
        logging.debug("genotype file not found: {}".format(path))
        return Genotype(dag=None, ops=None)
    with open(path, 'r', encoding='UTF-8') as f:
        g_str = f.read()
    return from_str(g_str)

def from_str(s):
    """ generate genotype from string """
    genotype = eval(s)
    return genotype

def get_genotype(g_str, ovr_genotype):
    if not ovr_genotype is None:
        genotype = from_file(ovr_genotype)
    elif g_str == '':
        genotype = from_file(config.gt_file)
    else:
        genotype = from_str(g_str)
    return genotype
