#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import copy
import argparse
import traceback

import combo_nas.utils as utils
from combo_nas.utils.config import Config
from combo_nas.utils.routine import search
from combo_nas.utils.wrapper import init_all_search
from combo_nas.hparam import build_hparam_space, HParamSpace, tune
from combo_nas.arch_optim import build_arch_optim

trial_index = 0

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

    build_hparam_space('hparams.json')
    optim_kwargs = dict(config.tune.copy())
    del optim_kwargs['type']
    optim_kwargs = config.tune.get('args', optim_kwargs)
    optim = build_arch_optim(config.tune.type, space=HParamSpace, **optim_kwargs)
    optim_batch_size = config.get('batch_size', 1)

    def measure(hp):
        global trial_index
        trial_config = copy.deepcopy(config)
        Config.apply(trial_config, hp)
        trial_name = '{}_{}'.format(args.name, trial_index)
        exp_root_dir = 'exp'
        trial_index += 1
        try:
            search_kwargs = init_all_search(trial_config, trial_name, exp_root_dir, args.chkpt, args.device, convert_fn=None)
            best_top1, best_gt, gts = search(**search_kwargs)
            score = best_top1
            error_no = 0
        except Exception as e:
            traceback.print_exc()
            score = 0
            error_no = 1
            logging.debug('trial {} failed with exit code: {}'.format(trial_index, error_no))
            
        result = {
            'score': score,
            'error_no': error_no,
        }
        return result

    tune(optim, measure, n_trial=2000, bsize=optim_batch_size, early_stopping=100)


if __name__ == '__main__':
    main()