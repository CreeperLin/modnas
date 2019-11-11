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
from combo_nas.hparam import build_hparam_tuner, build_hparam_space

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

    hp_space = build_hparam_space('hparams.json')
    tuner = build_hparam_tuner(config.tune.tuner, hp_space)


    def measure(hp):
        global trial_index
        trial_config = copy.deepcopy(config)
        Config.apply(trial_config, hp)
        trial_name = '{}_{}'.format(args.name, trial_index)
        exp_root_dir = os.path.join('exp', trial_name)
        trial_index += 1
        try:
            search_kwargs = init_all_search(trial_config, trial_name, exp_root_dir, args.device, convert_fn=None)
            best_top1, best_gt, gts = search(trial_config.search, args.chkpt, **search_kwargs)
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

    tuner.tune(measure, n_trial=2000, early_stopping=100)


if __name__ == '__main__':
    main()