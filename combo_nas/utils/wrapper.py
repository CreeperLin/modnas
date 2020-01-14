import os
from ..utils.exp_manager import ExpManager
from ..data_provider.dataloader import load_data
from ..arch_space.constructor import convert_from_predefined_net
from ..arch_space.constructor import convert_from_genotype
from ..arch_space.constructor import convert_from_layers
from ..arch_space.ops import configure_ops
from ..arch_space import build_arch_space
from ..arch_space.constructor import Slot
from ..core.param_space import ArchParamSpace
from ..core.controller import NASController
from ..arch_optim import build_arch_optim
from .. import utils as utils
from ..utils.config import Config
from ..arch_space import genotypes as gt
from ..hparam.space import build_hparam_space_from_dict, HParamSpace
from ..metrics import build_metrics
from .routine import search, augment, hptune

def load_config(conf, excludes):
    if isinstance(conf, Config):
        config = conf
    else:
        config = Config(conf)
    if utils.check_config(config, excludes):
        raise Exception("config error.")
    return config

def init_all_search(config, name, exp, chkpt, device, genotype=None, convert_fn=None):
    trn_loader = val_loader = model_builder = model = None
    ArchParamSpace.reset()
    config = load_config(config, excludes=['augment'])
    # dir
    expman = ExpManager(exp, name)
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    if 'data' in config.search:
        sp_ratio = config.search.data.dloader.split_ratio
        if sp_ratio > 0:
            trn_loader, val_loader = load_data(config.search.data, validation=False)
        else:
            trn_loader = load_data(config.search.data, validation=False)
            val_loader = None
    # ops
    if 'ops' in config:
        configure_ops(config.ops)
    # model
    if 'model' in config:
        def default_model_builder(genotype=genotype, convert_fn=convert_fn):
            Slot.reset()
            # net
            net = build_arch_space(config.model.type, config.model)
            # layers
            if not isinstance(convert_fn, list):
                convert_fn = [convert_fn]
            layer_convert_fn = convert_fn[:-1]
            layers_conf = config.get('layers', None)
            if not layers_conf is None:
                convert_from_layers(net, layers_conf, layer_convert_fn)
            fn_kwargs = {}
            # mixed_op
            if 'mixed_op' in config:
                # primitives
                if 'primitives' in config.mixed_op:
                    fn_kwargs['ops'] = config.mixed_op.primitives
                mixed_op_args = config.mixed_op.get('args', {})
                fn_kwargs.update(mixed_op_args)
                fn_kwargs['rid'] = config.mixed_op.type
            op_convert_fn = convert_fn[-1]
            if genotype is None:
                if op_convert_fn is None and hasattr(net, 'get_predefined_search_converter'):
                    op_convert_fn = net.get_predefined_search_converter()
                convert_from_predefined_net(net, op_convert_fn, fn_kwargs=fn_kwargs)
            else:
                if op_convert_fn is None and hasattr(net, 'get_genotype_search_converter'):
                    op_convert_fn = net.get_genotype_search_converter()
                genotype = gt.get_genotype(config.genotypes, genotype)
                fn_kwargs.update(config.genotypes.build_args)
                convert_from_genotype(net, genotype, op_convert_fn, fn_kwargs=fn_kwargs)
            # metrics
            metrics = None
            if 'metrics' in config:
                metrics = build_metrics(config.metrics.type, **config.metrics.get('args', {}))
            # controller
            crit = utils.get_net_crit(config.criterion)
            model = NASController(net, crit, metrics, dev_list).to(device=dev)
            # genotype
            if config.genotypes.disable_dag:
                model.to_genotype = model.to_genotype_ops
            if config.genotypes.use_slot:
                model.to_genotype_ops = model.to_genotype_slots
            model.to_genotype_args = config.genotypes.to_args
            # init
            model.init_model(config.init)
            return model
        model = default_model_builder()
        model_builder = default_model_builder
    # arch
    arch_kwargs = dict(config.arch_optim.copy())
    del arch_kwargs['type']
    arch_kwargs = config.arch_optim.get('args', arch_kwargs)
    arch = build_arch_optim(config.arch_optim.type, space=ArchParamSpace, **arch_kwargs)
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.search,
        'chkpt_path': chkpt,
        'arch_optim': arch,
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


def init_all_augment(config, name, exp, chkpt, device, genotype, convert_fn=None):
    config = load_config(config, excludes=['search'])
    # dir
    expman = ExpManager(exp, name)
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    val_loader = load_data(config.augment.data, validation=True)
    trn_loader = load_data(config.augment.data, validation=False)
    # ops
    if 'ops' in config:
        config.ops.affine = True
        configure_ops(config.ops)
    # net
    def model_builder(genotype=genotype, convert_fn=convert_fn):
        Slot.reset()
        net = build_arch_space(config.model.type, config.model)
        # layers
        if not isinstance(convert_fn, list):
            convert_fn = [convert_fn]
        layer_convert_fn = convert_fn[:-1]
        layers_conf = config.get('layers', None)
        if not layers_conf is None:
            convert_from_layers(net, layers_conf, layer_convert_fn)
        fn_kwargs = {}
        # op
        op_convert_fn = convert_fn[-1]
        if genotype is None:
            if op_convert_fn is None and hasattr(net, 'get_predefined_augment_converter'):
                op_convert_fn = net.get_predefined_augment_converter()
            convert_from_predefined_net(net, op_convert_fn, fn_kwargs=fn_kwargs)
        else:
            if op_convert_fn is None and hasattr(net, 'get_genotype_augment_converter'):
                op_convert_fn = net.get_genotype_augment_converter()
            genotype = gt.get_genotype(config.genotypes, genotype)
            fn_kwargs.update(config.genotypes.build_args)
            convert_from_genotype(net, genotype, op_convert_fn, fn_kwargs=fn_kwargs)
        # controller
        crit = utils.get_net_crit(config.criterion)
        model = NASController(net, crit, None, dev_list).to(device=dev)
        # init
        model.init_model(config.init)
        return model
    model = model_builder()
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.augment,
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


def init_all_hptune(config, name, exp, chkpt, device, measure_fn=None):
    HParamSpace.reset()
    config = load_config(config, excludes=['search', 'augment'])
    # dir
    expman = ExpManager(exp, name)
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    # device
    dev, _ = utils.init_device(config.device, device)
    # hpspace
    hp_path = config.hpspace.get('hp_path', None)
    if hp_path is None:
        build_hparam_space_from_dict(config.hpspace.hp_dict)
    # optim
    optim_kwargs = dict(config.hptuner.copy())
    del optim_kwargs['type']
    optim_kwargs = config.hptuner.get('args', optim_kwargs)
    optim = build_arch_optim(config.hptuner.type, space=HParamSpace, **optim_kwargs)
    # measure_fn
    if measure_fn is None:
        measure_fn = default_measure_fn
    return {
        'config': config.hptune,
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
    if proc == 'search':
        return default_search_measure_fn(*args, **kwargs)
    elif proc == 'augment':
        return default_augment_measure_fn(*args, **kwargs)
    elif proc == 'hptune':
        return default_hptune_measure_fn(*args, **kwargs)


def default_search_measure_fn(*args, **kwargs):
    best_top1, _, _ = run_search(*args, **kwargs)
    return best_top1


def default_augment_measure_fn(*args, **kwargs):
    best_top1 = run_augment(*args, **kwargs)
    return best_top1


def default_hptune_measure_fn(*args, **kwargs):
    _, _, best_hparams = run_hptune(*args, **kwargs)
    return best_hparams


def run_search(*args, **kwargs):
    return search(**init_all_search(*args, **kwargs))


def run_augment(*args, **kwargs):
    return augment(**init_all_augment(*args, **kwargs))


def run_hptune(*args, **kwargs):
    return hptune(**init_all_hptune(*args, **kwargs))
