#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from combo_nas.utils.wrapper import run_hptune

def main():
    parser = argparse.ArgumentParser(description='ComboNAS hparam tuning routine')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c', '--config', action='append', type=str, required=True,
                        help="yaml config file")
    parser.add_argument('-e', '--exp', type=str, default='exp',
                        help="experiment root dir")
    parser.add_argument('-d', '--device', type=str, default="all",
                        help="override device ids")
    parser.add_argument('-o', '--config_override', action='append', type=str, default=None,
                        help="override config")
    args = parser.parse_args()

    run_hptune(**vars(args))


if __name__ == '__main__':
    main()
