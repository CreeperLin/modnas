"""Run hyperparameter tuning on python scripts."""
import os
import sys
import yaml
import argparse
from .func import tune


def tune_script():
    """Run hyperparameter tuning on python scripts."""
    parser = argparse.ArgumentParser(description='Tune script hyperaparameters')
    parser.add_argument('-n', '--name', default=None, help="name of the experiment")
    parser.add_argument('-f', '--func', default=None, help="name of the tuned function")
    parser.add_argument('-c', '--config', default=None, help="yaml config file")
    parser.add_argument('-p', '--hparam', default=None, help="hparam config")
    parser.add_argument('-o', '--override', default=None, help="config override")
    (opts, args) = parser.parse_known_args()
    progname = args[0]
    funcname = opts.func
    hp_dict = {}
    if opts.hparam is not None:
        hp_dict = yaml.load(opts.hparam, Loader=yaml.FullLoader)
    sys.argv[:] = args[:]
    sys.path.insert(0, os.path.dirname(progname))
    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')
    globs = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }
    exec(code, globs, None)
    if funcname is None:
        for k in globs.keys():
            if not k.startswith('_'):
                funcname = k
                break
    func = globs.get(funcname, None)
    if func is None:
        raise ValueError('function {} not exist'.format(funcname))
    tune_name = opts.name or '{}.{}'.format(os.path.basename(progname), funcname)
    tune(func, *args[1:], tune_name=tune_name, tune_config=opts.config, tune_options=opts.override, tuned_args=hp_dict)


if __name__ == '__main__':
    tune_script()
