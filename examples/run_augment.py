#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import time
import torch
import argparse
from combo_nas.utils.routine import augment
from combo_nas.utils.wrapper import init_all_augment

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

    exp_root_dir = os.path.join('exp', args.name)
    augment_kwargs = init_all_augment(args.config, args.name, exp_root_dir, args.chkpt, args.device, args.genotype)
    augment(**augment_kwargs)


if __name__ == '__main__':
    main()