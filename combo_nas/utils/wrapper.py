import os
import queue
import importlib
from ..utils.exp_manager import ExpManager
from ..data_provider import load_data
from ..arch_space.constructor import convert_from_predefined_net,\
    convert_from_genotype, convert_from_layers, default_mixed_op_converter,\
    default_genotype_converter
from ..arch_space.ops import configure_ops
from ..arch_space import build as build_arch_space
from ..arch_space.constructor import Slot
from ..core.param_space import ArchParamSpace
from ..core.controller import NASController
from ..optim import build as build_optim
from .. import utils
from ..utils.config import Config, merge_dict
from ..arch_space import genotypes as gt
from ..hparam.space import build_hparam_space_from_dict, HParamSpace
from .routine import search, augment, hptune

def import_modules(modules):
    for m in modules:
        importlib.import_module(m)


def load_config(conf):
    if not isinstance(conf, list):
        conf = [conf]
    config = None
    for cfg in conf:
        loaded_cfg = Config.load(cfg)
        config = loaded_cfg if config is None else merge_dict(config, loaded_cfg)
    return config


def init_all_search(config, name, exp='exp', chkpt=None, device='all', genotype=None, convert_fn=None, config_override=None):
    model_builder = model = None
    ArchParamSpace.reset()
    # config
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.get('search', {}))
    utils.check_config(config, top_keys=['log', 'convert', 'genotypes', 'device'])
    # dir
    expman = ExpManager(exp, name, **config.log.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', []))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    trn_loader, val_loader = load_data(config, False)
    # ops
    if 'ops' in config:
        configure_ops(config.ops)
    # model
    if 'model' in config:
        def default_model_builder(genotype=genotype, convert_fn=convert_fn):
            Slot.reset()
            # net
            net = build_arch_space(config.model.type, **config.model.get('args', {}))
            # layers
            if not isinstance(convert_fn, list):
                convert_fn = [convert_fn]
            layer_convert_fn = convert_fn[:-1]
            layers_conf = config.get('layers', None)
            if not layers_conf is None:
                convert_from_layers(net, layers_conf, layer_convert_fn)
            fn_kwargs = {}
            # mixed_op
            use_mixed_op = False
            if 'mixed_op' in config:
                # primitives
                if 'primitives' in config.mixed_op:
                    fn_kwargs['primitives'] = config.mixed_op.primitives
                fn_kwargs['mixed_op_type'] = config.mixed_op.type
                fn_kwargs['mixed_op_args'] = config.mixed_op.get('args', {})
                use_mixed_op = True
            final_convert_fn = convert_fn[-1]
            if genotype is None:
                if final_convert_fn is None and hasattr(net, 'get_predefined_search_converter'):
                    final_convert_fn = net.get_predefined_search_converter(**config.convert.get('predefined_search_args', {}))
                if final_convert_fn is None and use_mixed_op:
                    final_convert_fn = default_mixed_op_converter
                convert_from_predefined_net(net, final_convert_fn, fn_kwargs=fn_kwargs)
            else:
                if final_convert_fn is None and hasattr(net, 'get_genotype_search_converter'):
                    final_convert_fn = net.get_genotype_search_converter(**config.convert.get('genotype_search_args', {}))
                if final_convert_fn is None:
                    final_convert_fn = default_genotype_converter
                genotype = gt.get_genotype(config.genotypes, genotype)
                convert_from_genotype(net, genotype, final_convert_fn, fn_kwargs=fn_kwargs)
            # controller
            if config.model.get('virtual', False):
                return net
            model = NASController(net, dev_list).to(device=dev)
            # genotype
            if config.genotypes.disable_dag:
                model.to_genotype = model.to_genotype_ops
            if config.genotypes.use_slot:
                model.to_genotype_ops = model.to_genotype_slots
            if config.genotypes.use_fallback:
                model.to_genotype_ops = model.to_genotype_fallback
            model.to_genotype_args = config.genotypes.to_args
            # init
            model.init_model(**config.get('init',{}))
            return model
        model = default_model_builder()
        model_builder = default_model_builder
    # optim
    optim_kwargs = config.optim.get('args', {})
    optim = build_optim(config.optim.type, space=ArchParamSpace, logger=logger, **optim_kwargs)
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.estimator,
        'chkpt_path': chkpt,
        'optim': optim,
        'estim_kwargs': {
            'expman': expman,
            'train_loader': trn_loader,
            'valid_loader': val_loader,
            'model_builder': model_builder,
            'model': model,
            'writer': writer,
            'logger': logger,
            'device': dev,
        }
    }


def init_all_augment(config, name, exp='exp', chkpt=None, device='all', genotype=None, convert_fn=None, config_override=None):
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.get('augment', {}))
    utils.check_config(config, top_keys=['log', 'convert', 'genotypes', 'device'])
    # dir
    expman = ExpManager(exp, name, **config.log.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', []))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    trn_loader, val_loader = load_data(config, True)
    # ops
    if 'ops' in config:
        config.ops.affine = True
        configure_ops(config.ops)
    # net
    def model_builder(genotype=genotype, convert_fn=convert_fn):
        Slot.reset()
        net = build_arch_space(config.model.type, **config.model.get('args', {}))
        # layers
        if not isinstance(convert_fn, list):
            convert_fn = [convert_fn]
        layer_convert_fn = convert_fn[:-1]
        layers_conf = config.get('layers', None)
        if not layers_conf is None:
            convert_from_layers(net, layers_conf, layer_convert_fn)
        # final
        final_convert_fn = convert_fn[-1]
        if genotype is None:
            if final_convert_fn is None and hasattr(net, 'get_predefined_augment_converter'):
                final_convert_fn = net.get_predefined_augment_converter(**config.convert.get('predefined_augment_args', {}))
            if final_convert_fn is None:
                raise ValueError('convert function required for augment run')
            convert_from_predefined_net(net, final_convert_fn)
        else:
            if final_convert_fn is None and hasattr(net, 'get_genotype_augment_converter'):
                final_convert_fn = net.get_genotype_augment_converter(**config.convert.get('genotype_augment_args', {}))
            if final_convert_fn is None:
                final_convert_fn = default_genotype_converter
            genotype = gt.get_genotype(config.genotypes, genotype)
            convert_from_genotype(net, genotype, final_convert_fn)
        # controller
        if config.model.get('virtual', False):
            return net
        model = NASController(net, dev_list).to(device=dev)
        # init
        model.init_model(**config.get('init',{}))
        return model
    model = model_builder()
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.estimator,
        'chkpt_path': chkpt,
        'estim_kwargs': {
            'expman': expman,
            'train_loader': trn_loader,
            'valid_loader': val_loader,
            'model_builder': model_builder,
            'model': model,
            'writer': writer,
            'logger': logger,
            'device': dev,
        }
    }


def init_all_hptune(config, name, exp='exp', chkpt=None, device='all', measure_fn=None, config_override=None):
    HParamSpace.reset()
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.get('hptune', {}))
    utils.check_config(config, top_keys=['log', 'device'])
    # dir
    expman = ExpManager(exp, name, **config.log.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', []))
    # device
    dev, _ = utils.init_device(config.device, device)
    # hpspace
    hp_path = config.hpspace.get('hp_path', None)
    if hp_path is None:
        build_hparam_space_from_dict(config.hpspace.hp_dict)
    # optim
    optim_kwargs = config.optim.get('args', {})
    optim = build_optim(config.optim.type, space=HParamSpace, logger=logger, **optim_kwargs)
    # measure_fn
    if measure_fn is None:
        measure_fn = default_measure_fn
    return {
        'config': config.estimator,
        'chkpt_path': chkpt,
        'optim': optim,
        'estim_kwargs': {
            'expman': expman,
            'train_loader': None,
            'valid_loader': None,
            'model_builder': None,
            'model': None,
            'writer': writer,
            'logger': logger,
            'device': dev,
            'measure_fn': measure_fn,
        }
    }


def default_measure_fn(proc, *args, **kwargs):
    runner = get_runner(proc)
    ret = runner(*args, **kwargs)
    if proc == 'search':
        return ret['best_top1']
    elif proc == 'augment':
        return ret['best_top1']
    elif proc == 'hptune':
        return ret['best_score']
    elif proc == 'pipeline':
        return ret['final']['best_top1']


def run_search(*args, **kwargs):
    return search(**init_all_search(*args, **kwargs))


def run_augment(*args, **kwargs):
    return augment(**init_all_augment(*args, **kwargs))


def run_hptune(*args, **kwargs):
    return hptune(**init_all_hptune(*args, **kwargs))


def run_pipeline(config, name, exp='exp', config_override=None):
    config = load_config(config)
    Config.apply(config, config_override or {})
    Config.apply(config, config.get('pipeline', {}))
    utils.check_config(config, top_keys=['log'])
    # dir
    expman = ExpManager(exp, name, **config.log.get('expman', {}))
    logger = utils.get_logger(expman.logs_path, name, config.log)
    # writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    logger.info(utils.env_info())
    # imports
    import_modules(config.get('imports', []))
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
            if not dep in finished:
                dep_sat = False
                break
        if not dep_sat:
            pending.put(pname)
            continue
        ptype = pconf.type
        proc = get_runner(ptype)
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


_runner_map = {
    'search': run_search,
    'augment': run_augment,
    'hptune': run_hptune,
    'pipeline': run_pipeline
}


def get_runner(rtype):
    if rtype in _runner_map:
        return _runner_map[rtype]
    else:
        raise ValueError('pipeline: unknown type: {}'.format(rtype))


def run(config, *args, proc=None, **kwargs):
    config = Config.load(config)
    proc = config.get('proc', None) if proc is None else proc
    proc = get_runner(proc)
    return proc(*args, config=config, **kwargs)
