#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from combo_nas.utils.wrapper import run_augment

def main():
    parser = argparse.ArgumentParser(description='ComboNAS augment routine')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml config file")
    parser.add_argument('-e', '--exp', type=str, default='exp',
                        help="experiment root dir")
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-d', '--device', type=str, default="all",
                        help="override device ids")
    parser.add_argument('-g', '--genotype', type=str, default=None,
                        help="override genotype file")
    args = parser.parse_args()

    run_augment(**vars(args))


if __name__ == '__main__':
    main()
