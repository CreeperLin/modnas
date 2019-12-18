#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
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

    exp_root_dir = 'exp'
    search_kwargs = init_all_search(args.config, args.name, exp_root_dir, args.chkpt, args.device, convert_fn=None)
    search(**search_kwargs)


if __name__ == '__main__':
    main()