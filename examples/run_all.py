#!/usr/bin/env python3
import argparse
from combo_nas.utils.wrapper import run_search
from combo_nas.utils.wrapper import run_augment

def main():
    parser = argparse.ArgumentParser(description='ComboNAS search & augment routine')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the model')
    parser.add_argument('-c', '--config', type=str, action='append', required=True,
                        help='yaml config file')
    parser.add_argument('-e', '--exp', type=str, default='exp',
                        help='experiment root dir')
    parser.add_argument('-p', '--chkpt', type=str, default=None,
                        help='path of checkpoint pt file')
    parser.add_argument('-d', '--device', type=str, default='all',
                        help='override device ids')
    parser.add_argument('-g', '--arch_desc', type=str, default=None,
                        help='override arch_desc file')
    parser.add_argument('-o', '--config_override', action='append', type=str, default=None,
                        help='override config')
    args = parser.parse_args()

    kwargs = vars(args)
    ret = run_search(**kwargs)
    kwargs['arch_desc'] = ret['final']['best_arch']
    run_augment(**kwargs)


if __name__ == '__main__':
    main()
