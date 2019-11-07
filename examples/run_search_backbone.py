#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import argparse

from model import *

import combo_nas.utils as utils
from combo_nas.utils.config import Config
from combo_nas.utils.routine import search
from combo_nas.utils.wrapper import init_all_search

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
        raise Exception("Config error.")
    conf_str = config.to_string()
    config.model.type = 'CustomNet'
    config.arch_optim.type = 'DARTS'
    config.mixed_op.type = 'DARTS'

    register_custom_ops()
    exp_root_dir = os.path.join('exp', args.name)
    search_kwargs = init_all_search(config, args.name, exp_root_dir, args.device, convert_fn=custom_backbone_cvt)
    search(config=config.search, chkpt_path=args.chkpt, **search_kwargs)


if __name__ == '__main__':
    main()