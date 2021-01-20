import copy
import logging
from functools import partial
from modnas.utils.wrapper import run_hptune

_default_hptune_config = {
    'optim': {
        'type': 'RandomSearchOptim'
    },
    'estimator': {
        'tune': {
            'type': 'HPTuneEstim',
            'epochs': -1,
        }
    }
}


def tune(func, *args, tune_config=None, tune_options=None, tuned_args=None, tuned=False, **kwargs):
    tuned_args = tuned_args or {}

    def parse_hp(hp):
        fn_kwargs = copy.deepcopy(kwargs)
        hp_kwargs = {k: v for k, v in hp.items() if not k.startswith('#')}
        fn_kwargs.update(hp_kwargs)
        fn_args = [hp.get('#{}'.format(i), v) for i, v in enumerate(args)]
        return fn_args, fn_kwargs

    def measure_fn(hp):
        fn_args, fn_kwargs = parse_hp(hp)
        return func(*fn_args, **fn_kwargs)

    opts = {
        'name': func.__name__,
        'config': [_default_hptune_config.copy()],
    }
    opts['config'][0]['hp_space'] = tuned_args
    opts['config'].extend(tune_config or [])
    opts['config_override'] = tune_options
    tune_res = run_hptune(measure_fn=measure_fn, **opts)
    best_hparams = list(tune_res.values())[0]['best_hparams']
    logging.info('tune: best hparams: {}'.format(dict(best_hparams)))
    ret = parse_hp(best_hparams)
    if tuned:
        fn_args, fn_kwargs = ret
        return func(*fn_args, **fn_kwargs)
    return ret


def wrap_tune(func, *dec_args, tune_config=None, tune_options=None, tuned_args=None, tuned=False, **dec_kwargs):
    tuned_args = tuned_args or {}
    tuned_args.update(dec_kwargs)
    tuned_args.update({'#{}'.format(i): v for i, v in enumerate(dec_args) if v is not None})

    return lambda *args, **kwargs: tune(
        func, *args, tune_config=tune_config, tune_options=tune_options, tuned_args=tuned_args, tuned=tuned, **kwargs)


def tune_func(*args, **kwargs):
    def tuner(func):
        return wrap_tune(func, *args, **kwargs)

    return tuner


tuned = partial(tune, tuned=True)
wrap_tuned = partial(wrap_tune, tuned=True)
tuned_func = partial(tune_func, tuned=True)
