#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import torch
import logging
import argparse
from utils.exp_manager import ExpManager

from utils.routine import augment
from utils.config import Config
import utils
import arch_space.genotypes as gt
from arch_space.constructor import convert_from_genotype
from arch_space import build_arch_space
from core.nas_modules import build_nas_controller
from data_provider.dataloader import load_data

from model import *

def main():
    parser = argparse.ArgumentParser(description='Proxyless-NAS augment')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c','--config',type=str, default='./config/default.yaml',
                        help="yaml config file")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-d','--device',type=str,default="all",
                        help="override device ids")
    parser.add_argument('-g','--genotype',type=str,default=None,
                        help="override genotype file")
    args = parser.parse_args()

    config = Config(args.config)
    if utils.check_config(config, args.name):
        raise Exception("Config error.")
    
    exp_root_dir = os.path.join('exp', args.name)
    expman = ExpManager(exp_root_dir)

    logger = utils.get_logger(expman.logs_path, args.name)
    writer = utils.get_writer(expman.writer_path, config.log.writer)

    dev, dev_list = utils.init_device(config.device, args.device)

    trn_loader = load_data(config.augment.data, validation=False)
    val_loader = load_data(config.augment.data, validation=True)

    gt.set_primitives(config.primitives)

    # load genotype
    genotype = gt.get_genotype(config.augment.genotype, args.genotype)
    
    # net = CustomBackbone(config.model.channel_in)
    net = build_arch_space(config.model.type, config.model)
    # convert_fn = custom_genotype_cvt
    # convert_fn = custom_genotype_space_cvt
    supernet = convert_from_genotype(net, genotype)

    model = build_nas_controller(config.model, supernet, dev, dev_list)

    augment(expman, args.chkpt, trn_loader, val_loader, model, writer, logger, dev, config.augment)


if __name__ == '__main__':
    main()