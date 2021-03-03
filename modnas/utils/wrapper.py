"""Wrapper for routine initialization and execution."""
import argparse
from collections import OrderedDict
from functools import partial
from ..registry.runner import build, register_as
from .exp_manager import ExpManager
from .config import Config
from ..core.event import EventManager
from ..core.param_space import ParamSpace
from ..registry.construct import build as build_con
from ..registry.callback import build as build_callback
from ..registry.export import build as build_exp
from ..registry.optim import build as build_optim
from ..registry.estim import build as build_estim
from ..registry.trainer import build as build_trainer
from .. import utils
from .logging import configure_logging, get_logger
from ..backend import use as use_backend
from . import predefined


logger = get_logger()


_default_arg_specs = [
    {
        'flags': ['-c', '--config'],
        'type': str,
        'required': True,
        'action': 'append',
        'help': 'yaml config file'
    },
    {
        'flags': ['-n', '--name'],
        'type': str,
        'default': None,
        'help': 'name of the job to run'
    },
    {
        'flags': ['-r', '--routine'],
        'type': str,
        'default': None,
        'help': 'routine type'
    },
    {
        'flags': ['-b', '--backend'],
        'type': str,
        'default': 'torch',
        'help': 'backend type'
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


DEFAULT_CALLBACK_CONF = ['ETAReporter', 'EstimReporter', 'TrainerReporter']


def parse_routine_args(name='default', arg_specs=None):
    """Return default arguments."""
    parser = argparse.ArgumentParser(description='ModularNAS {} routine'.format(name))
    arg_specs = arg_specs or _default_arg_specs
    for spec in arg_specs:
        parser.add_argument(*spec.pop('flags'), **spec)
    return vars(parser.parse_args())


def load_config(conf):
    """Load configurations."""
    if not isinstance(conf, list):
        conf = [conf]
    config = None
    for cfg in conf:
        loaded_cfg = Config.load(cfg)
        config = loaded_cfg if config is None else utils.merge_config(config, loaded_cfg)
    return config


def get_data_provider_config(config):
    keys = ['data', 'train_data', 'valid_data', 'data_loader', 'data_provider']
    return {k: config[k] for k in keys if k in config}


def get_init_constructor(config, device):
    """Return default init constructor."""
    default_conf = {'type': 'DefaultInitConstructor', 'args': {'device': device}}
    default_conf.update(config)
    return default_conf


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
        default_args['primitives'] = config.pop('primitives')
    default_args['mixed_op'] = config
    return {'type': default_type, 'args': default_args}


def get_arch_desc_constructor(arch_desc):
    """Return default archdesc constructor."""
    default_con = {'type': 'DefaultSlotArchDescConstructor', 'args': {}}
    default_con['args']['arch_desc'] = arch_desc
    return default_con


def build_constructor_all(config):
    """Build and return all constructors."""
    return OrderedDict([(k, build_con(conf)) for k, conf in config.items()])


def build_exporter_all(config):
    """Build and return all exporters."""
    if len(config) == 0:
        config = {'default': {'type': 'DefaultSlotTraversalExporter'}}
    if len(config) > 1:
        return build_exp('MergeExporter', config)
    if len(config) == 1:
        conf = list(config.values())[0]
        return build_exp(conf)
    return None


def build_trainer_all(trainer_config, trainer_comp=None):
    """Build and return all trainers."""
    trners = {}
    for trner_name, trner_conf in trainer_config.items():
        trner = build_trainer(trner_conf, **(trainer_comp or {}))
        trners[trner_name] = trner
    return trners


def build_estim_all(estim_config, estim_comp=None):
    """Build and return all estimators."""
    estims = {}
    if isinstance(estim_config, list):
        estim_config = OrderedDict([(c.get('name', str(i)), c) for i, c in enumerate(estim_config)])
    for estim_name, estim_conf in estim_config.items():
        estim = build_estim(estim_conf, name=estim_name, config=estim_conf, **(estim_comp or {}))
        estim.load(estim_conf.get('chkpt', None))
        estims[estim_name] = estim
    return estims


def bind_trainer(estims, trners):
    """Bind estimators with trainers."""
    for estim in estims.values():
        estim.set_trainer(trners.get(estim.config.get('trainer', estim.name), trners.get('default')))


def reset_all():
    ParamSpace().reset()
    EventManager().reset()


def estims_routine(optim, estims):
    """Run a chain of estimator routines."""
    results, ret = {}, None
    for estim_name, estim in estims.items():
        logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
        ret = estim.run(optim)
        results[estim_name] = ret
    logger.info('All results: {{\n{}\n}}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in results.items()])))
    results['final'] = ret
    reset_all()
    return results


def default_constructor(model, construct_fn=None, construct_config=None, arch_desc=None):
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
             backend='torch',
             chkpt=None,
             device=None,
             arch_desc=None,
             construct_fn=None,
             config_override=None,
             model=None):
    """Initialize all components from config."""
    if backend is not None:
        use_backend(backend)
    config = load_config(config)
    Config.apply(config, config_override or {})
    if routine:
        Config.apply(config, config.pop(routine, {}))
    utils.check_config(config)
    # dir
    name = name or utils.get_exp_name(config)
    expman = ExpManager(name, **config.get('expman', {}))
    configure_logging(config=config.get('logging', None), log_dir=expman.subdir('logs'))
    writer = utils.get_writer(expman.subdir('writer'), **config.get('writer', {}))
    logger.info('Name: {} Routine: {} Config:\n{}'.format(name, routine, config))
    logger.info(utils.env_info())
    # imports
    utils.import_modules(config.get('imports', None))
    # device
    device_conf = config.get('device', {})
    if device is not None:
        device_conf['device'] = device
    else:
        device = device_conf.get('device', device)
    # data
    data_provider_conf = get_data_provider_config(config)
    # construct
    con_config = OrderedDict()
    con_config['init'] = get_init_constructor(config.get('init', {}), device)
    if 'ops' in config:
        con_config['init']['args']['ops_conf'] = config.ops
    if 'model' in config:
        con_config['model'] = get_model_constructor(config.model)
    if 'mixed_op' in config:
        con_config['mixed_op'] = get_mixed_op_constructor(config.mixed_op)
    if arch_desc is not None:
        con_config['arch_desc'] = get_arch_desc_constructor(arch_desc)
    con_config = utils.merge_config(con_config, config.get('construct', {}))
    con_config['device'] = {'type': 'ToDevice', 'args': device_conf}
    if chkpt is not None:
        con_config['chkpt'] = get_chkpt_constructor(chkpt)
    # model
    constructor = partial(default_constructor,
                          construct_fn=construct_fn,
                          construct_config=con_config,
                          arch_desc=arch_desc)
    model = constructor(model)
    # export
    exporter = build_exporter_all(config.get('export', {}))
    # callback
    cb_config = config.get('callback', [])
    if isinstance(cb_config, dict):
        cb_config = [v for v in cb_config.values()]
    if 'NO_DEFAULT' not in cb_config:
        cb_config = DEFAULT_CALLBACK_CONF + cb_config
    for cb in cb_config:
        build_callback(cb)
    # optim
    optim = None
    if 'optim' in config:
        optim = build_optim(config.optim)
    # trainer
    trner_comp = {
        'data_provider': data_provider_conf,
        'writer': writer,
    }
    trners = build_trainer_all(config.get('trainer', {}), trner_comp)
    # estim
    estim_comp = {
        'expman': expman,
        'constructor': constructor,
        'exporter': exporter,
        'model': model,
        'writer': writer,
    }
    estims = build_estim_all(config.get('estim', {}), estim_comp)
    bind_trainer(estims, trners)
    return {'optim': optim, 'estims': estims}


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
    hptune_config = list(config.estim.values())[0]
    hptune_args = hptune_config.get('args', {})
    hptune_args['measure_fn'] = measure_fn
    hptune_config['args'] = hptune_args
    return init_all(config, *args, **kwargs)


def init_all_pipeline(config, *args, config_override=None, **kwargs):
    """Initialize all components from pipeline config."""
    config = load_config(config)
    Config.apply(config, config_override or {})
    config_override = {'estim': {'pipeline': {'type': 'PipelineEstim', 'pipeline': config.get('pipeline', {})}}}
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
        kwargs = parse_routine_args()
    routine_parsed = kwargs.pop('routine', None) or 'default'
    return build(routine or routine_parsed, *args, **kwargs)
