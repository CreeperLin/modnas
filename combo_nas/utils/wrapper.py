import os
import sys
import queue
import importlib
from collections import OrderedDict
from functools import partial
from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('runner')
from ..utils.exp_manager import ExpManager
from ..data_provider import get_data_provider
from ..arch_space.construct import build as build_con
from ..arch_space.export import build as build_exp
from ..arch_space.ops import configure_ops
from ..arch_space.slot import Slot
from ..core.param_space import ArchParamSpace
from ..optim import build as build_optim
from ..estim import build as build_estim
from ..trainer import build as build_trainer
from .. import utils
from ..utils.config import Config
from ..hparam.space import build_hparam_space_from_dict, HParamSpace


def import_modules(modules):
    if modules is None:
        return
    for m in modules:
        importlib.import_module(m)


def import_files(names, files):
    for name, path in zip(names, files):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def load_config(conf):
    if not isinstance(conf, list):
        conf = [conf]
    config = None
    for cfg in conf:
        loaded_cfg = Config.load(cfg)
        config = loaded_cfg if config is None else Config.merge(config, loaded_cfg)
    return config


def get_model_constructor(config):
    default_type = 'DefaultModelConstructor'
    default_args = {}
    default_args['model_type'] = config['type']
    if 'args' in config:
        default_args['args'] = config['args']
    return {'type': default_type, 'args': default_args}


def get_chkpt_constructor(path):
    return {'type': 'DefaultTorchCheckpointLoader', 'args': {'path': path}}


def get_mixed_op_constructor(config):
    default_type = 'DefaultMixedOpConstructor'
    default_args = {}
    if 'primitives' in config:
        default_args['primitives'] = config['primitives']
    if 'type' in config:
        default_args['mixed_type'] = config['type']
    if 'args' in config:
        default_args['mixed_args'] = config['args']
    return {'type': default_type, 'args': default_args}


def build_constructor_all(config):
    return OrderedDict([(k, build_con(conf['type'], **conf.get('args', {}))) for k, conf in config.items()])


def build_exporter_all(config):
    if len(config) == 0:
        config = {'default': {'type': 'DefaultSlotTraversalExporter'}}
    if len(config) > 1:
        return build_exp('MergeExporter', config)
    if len(config) == 1:
        conf = list(config.values())[0]
        return build_exp(conf['type'], **conf.get('args', {}))
    return None


def build_trainer_all(trainer_config, trainer_comp=None):
    trners = {}
    for trner_name, trner_conf in trainer_config.items():
        if isinstance(trner_conf, str):
            trner_conf = {'type': trner_conf}
        trner_args = trner_conf.get('args', {})
        trner_args.update(trainer_comp or {})
        trner = build_trainer(trner_conf['type'], **trner_args)
        trners[trner_name] = trner
    return trners


def build_estim_all(estim_config, estim_comp=None):
    estims = {}
    if isinstance(estim_config, list):
        estim_config = OrderedDict([(c.get('name', str(i)), c) for i, c in enumerate(estim_config)])
    for estim_name, estim_conf in estim_config.items():
        estim_type = estim_conf.type
        estim_args = estim_conf.get('args', {})
        estim_args.update(estim_comp or {})
        estim_args['name'] = estim_name
        estim_args['config'] = estim_conf
        estim = build_estim(estim_type, **estim_args)
        estim.load(estim_conf.get('chkpt', None))
        estims[estim_name] = estim
    return estims


def bind_trainer(estims, trners):
    for estim in estims.values():
        estim.set_trainer(trners.get(estim.config.get('trainer', estim.name), trners.get('default')))


def estims_routine(logger, optim, estims):
    results = {}
    for estim_name, estim in estims.items():
        logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
        ret = estim.run(optim)
        results[estim_name] = ret
        logger.info('Results: {}: {{{}}}'.format(estim_name, ', '.join(['{}: {}'.format(k, v) for k, v in ret.items()])))
    logger.info('All results: {{\n{}\n}}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in results.items()])))
    results['final'] = ret
    return results


def default_constructor(logger, construct_fn):
    Slot.reset()
    # net
    net = None
    for name, con_fn in construct_fn.items():
        logger.info('Running constructor: {} type: {}'.format(name, con_fn.__class__.__name__))
        net = con_fn(net)
    return net


def init_all(config, name, exp, chkpt, device, arch_desc, construct_fn):
    # reset
    ArchParamSpace.reset()
    # dir
    utils.check_config(config)
    expman = ExpManager(exp, name, **config.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, **config.get('logger', {}))
    writer = utils.get_writer(expman.writer_path, **config.get('writer', {}))
    logger.info('config loaded:\n{}'.format(config))
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
    if 'ops' in config:
        configure_ops(config.ops)
    # construct
    if construct_fn is None:
        construct_fn = {}
    if isinstance(construct_fn, list):
        construct_fn = [(str(i), v) for i, v in enumerate(construct_fn)]
    construct_fn = OrderedDict(construct_fn)
    con_config = OrderedDict()
    if 'model' in config:
        con_config['model'] = get_model_constructor(config.model)
    con_config.update(config.get('construct', {}))
    if 'mixed_op' in config:
        con_config['mixed_op'] = get_mixed_op_constructor(config.mixed_op)
    if arch_desc is not None:
        default_con = con_config.get('arch_desc', {'type': 'DefaultSlotArchDescConstructor'})
        args = default_con.get('args', {})
        args['arch_desc'] = arch_desc
        default_con['args'] = args
        con_config['arch_desc'] = default_con
    if device_ids and len(con_config) or len(construct_fn):
        con_config['device'] = {'type': 'ToDevice', 'args': {'device_ids': device_ids}}
    if chkpt is not None:
        con_config['chkpt'] = get_chkpt_constructor(chkpt)
    construct_fn.update(build_constructor_all(con_config))
    # model
    model = constructor = None
    if construct_fn:
        constructor = partial(default_constructor, logger, construct_fn)
        model = constructor()
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


def init_all_search(config, name, exp='exp', chkpt=None, device=None, arch_desc=None, construct_fn=None, config_override=None):
    # config
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.pop('search', {}))
    return init_all(config, name, exp, chkpt, device, arch_desc, construct_fn)


def init_all_augment(config,
                     name,
                     exp='exp',
                     chkpt=None,
                     device=None,
                     arch_desc=None,
                     construct_fn=None,
                     config_override=None):
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.pop('augment', {}))
    return init_all(config, name, exp, chkpt, device, arch_desc, construct_fn)


def init_all_hptune(config, name, exp='exp', measure_fn=None, config_override=None):
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.pop('hptune', {}))
    # hpspace
    HParamSpace.reset()
    build_hparam_space_from_dict(config.hpspace.get('hp_dict', {}))
    hptune_config = list(config.estimator.values())[0]
    hptune_args = hptune_config.get('args', {})
    hptune_args['measure_fn'] = measure_fn
    hptune_config['args'] = hptune_args
    return init_all(config, name, exp, None, None, None, None)


@register_as('search')
def run_search(*args, **kwargs):
    return estims_routine(**init_all_search(*args, **kwargs))


@register_as('augment')
def run_augment(*args, **kwargs):
    return estims_routine(**init_all_augment(*args, **kwargs))


@register_as('hptune')
def run_hptune(*args, **kwargs):
    return estims_routine(**init_all_hptune(*args, **kwargs))


@register_as('pipeline')
def run_pipeline(config, name, exp='exp', config_override=None):
    config = load_config(config)
    Config.apply(config, config_override or {})
    utils.check_config(config, top_keys=['log'])
    # dir
    expman = ExpManager(exp, name, **config.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, **config.get('logger', {}))
    logger.info('config loaded:\n{}'.format(config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', None))
    # pipeline
    pipeconf = config.pipeline
    pending = queue.Queue()
    for pn in pipeconf.keys():
        pending.put(pn)
    finished = set()
    ret_values = dict()
    while not pending.empty():
        pname = pending.get()
        pconf = pipeconf.get(pname)
        dep_sat = True
        for dep in pconf.get('depends', []):
            if dep not in finished:
                dep_sat = False
                break
        if not dep_sat:
            pending.put(pname)
            continue
        ptype = pconf.type
        proc = get_builder(ptype)
        pargs = pconf.get('args', {})
        pargs.exp = os.path.join(expman.root_dir, pargs.get('exp', ''))
        pargs.name = pargs.get('name', pname)
        for inp_kw, inp_idx in pconf.get('inputs', {}).items():
            keys = inp_idx.split('.')
            inp_val = ret_values
            for k in keys:
                inp_val = inp_val[k]
            pargs[inp_kw] = inp_val
        logger.info('pipeline: running {}, type={}'.format(pname, ptype))
        ret = proc(**pargs)
        ret_values[pname] = ret
        logger.info('pipeline: finished {}, results={}'.format(pname, ret))
        finished.add(pname)
    ret_values['final'] = ret
    logger.info('pipeline: all finished')
    return ret_values


def run(config, *args, proc=None, **kwargs):
    config = Config.load(config)
    proc = proc or config.get('proc', None)
    return build(proc, *args, config=config, **kwargs)
