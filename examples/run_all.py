#!/usr/bin/env python3
from modnas.utils.wrapper import parse_routine_args, run_search, run_augment


def main():
    kwargs = vars(parse_routine_args('search & augment').parse_args())
    ret = run_search(**kwargs)
    kwargs['arch_desc'] = ret['final']['best_arch']
    run_augment(**kwargs)


if __name__ == '__main__':
    main()
