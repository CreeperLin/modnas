#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from combo_nas.utils.wrapper import run_pipeline

def main():
    parser = argparse.ArgumentParser(description='ComboNAS pipeline')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml config file")
    parser.add_argument('-e', '--exp', type=str, default='exp',
                        help="experiment root dir")
    args = parser.parse_args()

    run_pipeline(**vars(args))


if __name__ == '__main__':
    main()
