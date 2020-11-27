"""Wrapper for routine initialization and execution."""
import importlib
import argparse
from collections import OrderedDict
from functools import partial
from ..registry.runner import register, get_builder, build, register_as
from .exp_manager import ExpManager
from .config import Config
from ..data_provider import get_data_provider
from ..arch_space.construct import build as build_con
from ..arch_space.export import build as build_exp
from ..arch_space.ops import configure_ops
from ..core.param_space import ArchParamSpace
from ..optim import build as build_optim
from ..estim import build as build_estim
from ..trainer import build as build_trainer
from .. import utils

_default_arg_specs = [
    {
        'flags': ['-n', '--name'],
        'type': str,
        'required': True,
        'help': 'name of the model'
    },
    {
        'flags': ['-r', '--routine'],
        'type': str,
        'default': None,
        'help': 'routine type'
    },
    {
        'flags': ['-c', '--config'],
        'type': str,
        'required': True,
        'action': 'append',
        'help': 'yaml config file'
    },
    {
        'flags': ['-p', '--chkpt'],
        'type': str,
        'default': None,
        'help': 'checkpoint file'
    },
    {
        'flags': ['-d', '--device'],
        'type': str,
        'default': 'all',
        'help': 'override device ids'
    },
    {
        'flags': ['-g', '--arch_desc'],
        'type': str,
        'default': None,
        'help': 'override arch_desc file'
    },
    {
        'flags': ['-o', '--config_override'],
        'type': str,
        'default': None,
        'help': 'override config',
        'action': 'append'
    },
]


def parse_routine_args(name='default', arg_specs=None):
    """Return default arguments."""
    parser = argparse.ArgumentParser(description='ModularNAS {} routine'.format(name))
    arg_specs = arg_specs or _default_arg_specs
    for spec in arg_specs:
        parser.add_argument(*spec.pop('flags'), **spec)
    return vars(parser.parse_args())


def import_modules(modules):
    """Import modules by name."""
    if modules is None:
        return
    if isinstance(modules, str):
        modules = [modules]
    for m in modules:
        importlib.import_module(m)


def load_config(conf):
    """Load configurations."""
    if not isinstance(conf, list):
        conf = [conf]
    config = None
    for cfg in conf:
        loaded_cfg = Config.load(cfg)
        config = loaded_cfg if config is None else utils.merge_config(config, loaded_cfg)
    return config


def get_init_constructor():
    """Return default init constructor."""
    return {'type': 'DefaultInitConstructor'}


def get_model_constructor(config):
    """Return default model constructor."""
    default_type = 'DefaultModelConstructor'
    default_args = {}
    default_args['model_type'] = config['type']
    if 'args' in config:
        default_args['args'] = config['args']
    return {'type': default_type, 'args': default_args}


def get_chkpt_constructor(path):
    """Return default checkpoint loader."""
    return {'type': 'DefaultTorchCheckpointLoader', 'args': {'path': path}}


def get_mixed_op_constructor(config):
    """Return default mixed operation constructor."""
    default_type = 'DefaultMixedOpConstructor'
    default_args = {}
    if 'primitives' in config:
        default_args['primitives'] = config['primitives']
    if 'type' in config:
        default_args['mixed_type'] = config['type']
    if 'args' in config:
        default_args['mixed_args'] = config['args']
    return {'type': default_type, 'args': default_args}


def get_arch_desc_constructor(arch_desc):
    """Return default archdesc constructor."""
    default_con = {'type': 'DefaultSlotArchDescConstructor', 'args': {}}
    default_con['args']['arch_desc'] = arch_desc
    return default_con


def build_constructor_all(config):
    """Build and return all constructors."""
    return OrderedDict([(k, build_con(conf['type'], **conf.get('args', {}))) for k, conf in config.items()])


def build_exporter_all(config):
    """Build and return all exporters."""
    if len(config) == 0:
        config = {'default': {'type': 'DefaultSlotTraversalExporter'}}
    if len(config) > 1:
        return build_exp('MergeExporter', config)
    if len(config) == 1:
        conf = list(config.values())[0]
        return build_exp(conf['type'], **conf.get('args', {}))
    return None


def build_trainer_all(trainer_config, trainer_comp=None):
    """Build and return all trainers."""
    trners = {}
    for trner_name, trner_conf in trainer_config.items():
        if isinstance(trner_conf, str):
            trner_conf = {'type': trner_conf}
        trner_args = trner_conf.get('args', {})
        trner = build_trainer(trner_conf['type'], **trner_args, **(trainer_comp or {}))
        trners[trner_name] = trner
    return trners


def build_estim_all(estim_config, estim_comp=None):
    """Build and return all estimators."""
    estims = {}
    if isinstance(estim_config, list):
        estim_config = OrderedDict([(c.get('name', str(i)), c) for i, c in enumerate(estim_config)])
    for estim_name, estim_conf in estim_config.items():
        estim_type = estim_conf['type']
        estim_args = estim_conf.get('args', {})
        estim_args.update(estim_comp or {})
        estim_args['name'] = estim_name
        estim_args['config'] = estim_conf
        estim = build_estim(estim_type, **estim_args)
        estim.load(estim_conf.get('chkpt', None))
        estims[estim_name] = estim
    return estims


def bind_trainer(estims, trners):
    """Bind estimators with trainers."""
    for estim in estims.values():
        estim.set_trainer(trners.get(estim.config.get('trainer', estim.name), trners.get('default')))


def estims_routine(logger, optim, estims):
    """Run a chain of estimator routines."""
    results, ret = {}, None
    for estim_name, estim in estims.items():
        logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
        ret = estim.run(optim)
        results[estim_name] = ret
    logger.info('All results: {{\n{}\n}}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in results.items()])))
    results['final'] = ret
    return results


def default_constructor(model, logger=None, construct_fn=None, construct_config=None, arch_desc=None):
    """Apply all constructors on model."""
    construct_fn = construct_fn or {}
    if isinstance(construct_fn, list):
        construct_fn = [(str(i), v) for i, v in enumerate(construct_fn)]
    construct_fn = OrderedDict(construct_fn)
    if arch_desc:
        construct_config['arch_desc']['args']['arch_desc'] = arch_desc
    construct_fn.update(build_constructor_all(construct_config or {}))
    for name, con_fn in construct_fn.items():
        logger.info('Running constructor: {} type: {}'.format(name, con_fn.__class__.__name__))
        model = con_fn(model)
    return model


def init_all(config,
             name=None,
             routine=None,
             chkpt=None,
             device=None,
             arch_desc=None,
             construct_fn=None,
             config_override=None,
             model=None):
    """Initialize all components from config."""
    config = load_config(config)
    Config.apply(config, config_override or {})
    if routine:
        Config.apply(config, config.pop(routine, {}))
    utils.check_config(config)
    # dir
    name = name or utils.get_exp_name(config)
    expman = ExpManager(name, **config.get('expman', {}))
    logger = utils.get_logger(expman.subdir('logs'), name, **config.get('logger', {}))
    writer = utils.get_writer(expman.subdir('writer'), **config.get('writer', {}))
    logger.info('routine: {} config:\n{}'.format(routine, config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', None))
    # device
    device_conf = config.get('device', {})
    if device:
        device_conf['device'] = device
    device, device_ids = utils.init_device(**device_conf)
    # data
    data_provider = get_data_provider(config, logger)
    # ops
    configure_ops(**config.get('ops', {}))
    # construct
    con_config = OrderedDict()
    con_config['init'] = get_init_constructor()
    if 'model' in config:
        con_config['model'] = get_model_constructor(config.model)
    if 'mixed_op' in config:
        con_config['mixed_op'] = get_mixed_op_constructor(config.mixed_op)
    if arch_desc is not None:
        con_config['arch_desc'] = get_arch_desc_constructor(arch_desc)
    con_config = utils.merge_config(con_config, config.get('construct', {}))
    if device_ids and len(con_config) > 1:
        con_config['device'] = {'type': 'ToDevice', 'args': {'device_ids': device_ids}}
    if chkpt is not None:
        con_config['chkpt'] = get_chkpt_constructor(chkpt)
    # model
    constructor = partial(default_constructor,
                          logger=logger,
                          construct_fn=construct_fn,
                          construct_config=con_config,
                          arch_desc=arch_desc)
    model = constructor(model)
    # export
    exporter = build_exporter_all(config.get('export', {}))
    # optim
    optim = None
    if 'optim' in config:
        optim_kwargs = config.optim.get('args', {})
        optim = build_optim(config.optim.type, space=ArchParamSpace, logger=logger, **optim_kwargs)
    # trainer
    trner_comp = {
        'data_provider': data_provider,
        'writer': writer,
        'logger': logger,
    }
    trners = build_trainer_all(config.get('trainer', {}), trner_comp)
    # estim
    estim_comp = {
        'expman': expman,
        'constructor': constructor,
        'exporter': exporter,
        'model': model,
        'writer': writer,
        'logger': logger,
    }
    estims = build_estim_all(config.get('estimator', {}), estim_comp)
    bind_trainer(estims, trners)
    return {'logger': logger, 'optim': optim, 'estims': estims}


def init_all_hptune(config, *args, config_override=None, measure_fn=None, **kwargs):
    """Initialize all components from hptune config."""
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.pop('hptune', {}))
    # hpspace
    if not config.get('construct', {}):
        config['construct'] = {
            'hparams': {
                'type': 'DefaultHParamSpaceConstructor',
                'args': {
                    'params': config.get('hp_space', {})
                }
            }
        }
    hptune_config = list(config.estimator.values())[0]
    hptune_args = hptune_config.get('args', {})
    hptune_args['measure_fn'] = measure_fn
    hptune_config['args'] = hptune_args
    return init_all(config, *args, **kwargs)


def init_all_pipeline(config, *args, config_override=None, **kwargs):
    """Initialize all components from pipeline config."""
    config = load_config(config)
    Config.apply(config, config_override or {})
    config_override = {
        'estimator': {
            'pipeline': {
                'type': 'PipelineEstim',
                'pipeline': config.get('pipeline', {})
            }
        }
    }
    return init_all(config, *args, config_override=config_override, **kwargs)


@register_as('default')
def run_default(*args, **kwargs):
    """Run search routines."""
    return estims_routine(**init_all(*args, **kwargs))


@register_as('search')
def run_search(*args, **kwargs):
    """Run search routines."""
    return estims_routine(**init_all(*args, routine='search', **kwargs))


@register_as('augment')
def run_augment(*args, **kwargs):
    """Run augment routines."""
    return estims_routine(**init_all(*args, routine='augment', **kwargs))


@register_as('hptune')
def run_hptune(*args, **kwargs):
    """Run hptune routines."""
    return estims_routine(**init_all_hptune(*args, **kwargs))


@register_as('pipeline')
def run_pipeline(*args, **kwargs):
    """Run pipeline routines."""
    return estims_routine(**init_all_pipeline(*args, **kwargs))


def run(*args, routine=None, **kwargs):
    """Run routine."""
    if not args and not kwargs:
        kwargs = parse_routine_args(name='default')
    routine = kwargs.pop('routine', 'default')
    return build(routine, *args, **kwargs)
