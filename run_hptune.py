#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import traceback
import utils
from utils.routine import search, augment
from utils.config import Config
from utils.exp_manager import ExpManager

from data_provider.dataloader import load_data
import arch_space.genotypes as gt
from arch_space.constructor import convert_from_predefined_net
from arch_space import build_arch_space
from core.nas_modules import build_nas_controller
from core.mixed_ops import build_mixed_op
from arch_optim import build_arch_optim
from hparam import build_hparam_tuner, build_hparam_space

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
    args = parser.parse_args()

    config = Config(args.config)
    if utils.check_config(config, args.name):
        raise Exception("config error.")
    config_str = config.to_string()

    exp_root_dir = os.path.join('exp', args.name)
    expman = ExpManager(exp_root_dir)

    logger = utils.get_logger(expman.logs_path, args.name)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    
    dev, dev_list = utils.init_device(config.device, args.device)

    trn_loader, val_loader = load_data(config.search.data, validation=False)

    gt.set_primitives(config.primitives)

    ops = gt.get_primitives()

    hp_space = build_hparam_space('hparams.json')
    tuner = build_hparam_tuner(config.tune.tuner, hp_space)

    def measure(hp):
        config.apply(hp)
        arch_space = config.model.type
        mixop_cls = config.model.mixed_op
        arch_optim = config.search.architect.type
        
        try:
            net = build_arch_space(arch_space, config.model)
            convert_fn = lambda slot: build_mixed_op(mixop_cls, slot.chn_in, slot.stride, ops, slot.param_pid)
            supernet = convert_from_predefined_net(net, convert_fn)

            model = build_nas_controller(config.model, supernet, dev, dev_list)
            arch = build_arch_optim(arch_optim, config.search, model)

            best_top1, best_gt, gts = search(expman, args.chkpt, trn_loader, val_loader, model, arch, writer, logger, dev, config.search)
            score = best_top1
            error_no = 0
        except Exception as e:
            traceback.print_exc()
            score = 0
            error_no = 1
            
        result = {
            'score': score,
            'error_no': error_no,
        }
        return result

    tuner.tune(measure, n_trial=2000, early_stopping=100)


if __name__ == '__main__':
    main()