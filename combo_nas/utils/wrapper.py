import os
import sys
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
from ..estimator import build as build_estimator
from .. import utils
from ..utils.config import Config
from ..arch_space import genotypes as gt
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


def build_estim_all(estim_config, estim_comp):
    estims = {}
    if isinstance(estim_config, list):
        estim_config = {c.get('name', str(i)): c for i, c in enumerate(estim_config)}
    for estim_name, estim_conf in estim_config.items():
        estim_type = estim_conf.type
        estim_args = estim_conf.get('args', {})
        estim_args.update(estim_comp)
        estim_args['name'] = estim_name
        estim_args['config'] = estim_conf
        estim = build_estimator(estim_type, **estim_args)
        estim.load(estim_conf.get('chkpt', None))
        estims[estim_name] = estim
    return estims


def estims_routine(logger, optim, estims):
    results = {}
    for estim_name, estim in estims.items():
        logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
        ret = estim.search(optim)
        results[estim_name] = ret
        logger.info('Results: {}: {{{}}}'.format(estim_name, ', '.join(['{}: {}'.format(k, v) for k, v in ret.items()])))
    logger.info('All results: {{\n{}\n}}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in results.items()])))
    results['final'] = ret
    return results


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
    import_modules(config.get('imports', None))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    data_provider = load_data(config, dev, False, logger)
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
            model = NASController(net, dev_list)
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
        model_builder = default_model_builder
        model = model_builder()
        if chkpt:
            model.load(chkpt)
    # optim
    optim = None
    if 'optim' in config:
        optim_kwargs = config.optim.get('args', {})
        optim = build_optim(config.optim.type, space=ArchParamSpace, logger=logger, **optim_kwargs)
    # estim
    estim_kwargs = {
        'expman': expman,
        'data_provider': data_provider,
        'model_builder': model_builder,
        'model': model,
        'writer': writer,
        'logger': logger,
    }
    estims = build_estim_all(config.estimator, estim_kwargs)
    return {
        'logger': logger,
        'optim': optim,
        'estims': estims
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
    import_modules(config.get('imports', None))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    data_provider = load_data(config, dev, True, logger)
    # ops
    if 'ops' in config:
        if not config.ops.get('affine', False):
            logger.warning('option \'ops.affine\' set to False in augment run')
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
        model = NASController(net, dev_list)
        # init
        model.init_model(**config.get('init',{}))
        return model
    model = model_builder()
    if chkpt:
        model.load(chkpt)
    # estim
    estim_kwargs = {
        'expman': expman,
        'data_provider': data_provider,
        'model_builder': model_builder,
        'model': model,
        'writer': writer,
        'logger': logger,
    }
    estims = build_estim_all(config.estimator, estim_kwargs)
    return {
        'logger': logger,
        'optim': None,
        'estims': estims
    }


def init_all_hptune(config, name, exp='exp', device='all', measure_fn=None, config_override=None):
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
    import_modules(config.get('imports', None))
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
    # estim
    estim_kwargs = {
        'expman': expman,
        'data_provider': None,
        'model_builder': None,
        'model': None,
        'writer': writer,
        'logger': logger,
        'measure_fn': measure_fn,
    }
    estims = build_estim_all(config.estimator, estim_kwargs)
    return {
        'logger': logger,
        'optim': optim,
        'estims': estims
    }


def default_measure_fn(proc, *args, **kwargs):
    runner = get_runner(proc)
    ret = runner(*args, **kwargs)
    ret = ret['final']
    return ret.get('best_score', list(ret.values())[0])


def run_search(*args, **kwargs):
    return estims_routine(**init_all_search(*args, **kwargs))


def run_augment(*args, **kwargs):
    return estims_routine(**init_all_augment(*args, **kwargs))


def run_hptune(*args, **kwargs):
    return estims_routine(**init_all_hptune(*args, **kwargs))


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
