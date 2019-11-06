#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import utils
from utils.routine import search
from utils.config import Config
from utils import param_count
from utils.exp_manager import ExpManager

from model import *

from data_provider.dataloader import load_data
import arch_space.genotypes as gt
from arch_space.constructor import convert_from_predefined_net, convert_from_genotype
from arch_optim import build_arch_optim
from arch_space import build_arch_space
from core.nas_modules import build_nas_controller
from core.mixed_ops import build_mixed_op


def main():
    parser = argparse.ArgumentParser()
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
    conf_str = config.to_string()

    exp_root_dir = os.path.join('exp', args.name)
    expman = ExpManager(exp_root_dir)

    logger = utils.get_logger(expman.logs_path, args.name)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    
    dev, dev_list = utils.init_device(config.device, args.device)

    trn_loader, val_loader = load_data(config.search.data, validation=False)

    gt.set_primitives(config.primitives)

    # register_custom_ops()

    genotype = gt.get_genotype(None, args.genotype)
    
    # net = CustomBackbone(config.model.channel_in)
    net = build_arch_space(config.model.type, config.model)
    # convert_fn = custom_genotype_cvt
    convert_fn = custom_genotype_space_cvt
    net = convert_from_genotype(net, genotype, convert_fn)

    model = build_nas_controller(config.model, net, dev, dev_list)
    arch = build_arch_optim('DARTS', config.search, model)

    search(expman, args.chkpt, trn_loader, val_loader, model, arch, writer, logger, dev, config.search)


if __name__ == '__main__':
    main()