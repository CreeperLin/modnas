#!/usr/bin/env python3
import argparse
from modnas.utils.wrapper import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='ModularNAS pipeline')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the model')
    parser.add_argument('-c', '--config', action='append', type=str, required=True,
                        help='yaml config file')
    parser.add_argument('-o', '--config_override', action='append', type=str, default=None,
                        help='override config')
    args = parser.parse_args()

    run_pipeline(**vars(args))


if __name__ == '__main__':
    main()
