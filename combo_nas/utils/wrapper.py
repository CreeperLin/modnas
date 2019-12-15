import os
from ..utils.exp_manager import ExpManager
from ..data_provider.dataloader import load_data
from ..arch_space.constructor import convert_from_predefined_net
from ..arch_space.constructor import convert_from_genotype
from ..arch_space.constructor import convert_from_layers
from ..arch_space.ops import configure_ops
from ..arch_space import build_arch_space
from ..arch_space.constructor import Slot
from ..core.nas_modules import ArchModuleSpace
from ..core.param_space import ArchParamSpace
from ..core.controller import NASController
from ..arch_optim import build_arch_optim
from .. import utils as utils
from ..utils.config import Config
from ..arch_space import genotypes as gt

def load_config(conf, name, excludes):
    if isinstance(conf, Config):
        config = conf
    else:
        config = Config(conf)
    if utils.check_config(config, name, excludes):
        raise Exception("config error.")
    return config

def init_all_search(config, name, exp_root_dir, chkpt, device, genotype=None, convert_fn=None):
    config = load_config(config, name, excludes=['augment'])
    # dir
    expman = ExpManager(exp_root_dir)
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    logger.info('config loaded:\n{}'.format(config))
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    sp_ratio = config.search.data.dloader.split_ratio
    if sp_ratio > 0:
        trn_loader, val_loader = load_data(config.search.data, validation=False)
    else:
        trn_loader = load_data(config.search.data, validation=False)
        val_loader = None
    # primitives
    gt.set_primitives(config.primitives)
    # net
    ArchParamSpace.reset()
    ArchModuleSpace.reset()
    Slot.reset()
    configure_ops(config.ops)
    net = build_arch_space(config.model.type, config.model)
    # layers
    if not isinstance(convert_fn, list):
        convert_fn = [convert_fn]
    layer_convert_fn = convert_fn[:-1]
    layers_conf = config.get('layers', None)
    if not layers_conf is None:
        convert_from_layers(net, layers_conf, layer_convert_fn)
    # mixed_op
    mixed_op_args = config.mixed_op.get('args', {})
    drop_path = config.search.drop_path_prob > 0.0
    op_convert_fn = convert_fn[-1]
    if genotype is None:
        if op_convert_fn is None and hasattr(net, 'get_predefined_search_converter'):
            op_convert_fn = net.get_predefined_search_converter()
        model = convert_from_predefined_net(net, op_convert_fn, drop_path, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
    else:
        if op_convert_fn is None and hasattr(net, 'get_genotype_search_converter'):
            op_convert_fn = net.get_genotype_search_converter()
        genotype = gt.get_genotype(config.genotypes, genotype)
        model = convert_from_genotype(net, genotype, op_convert_fn, drop_path, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
    # controller
    crit = utils.get_net_crit(config.criterion)
    model = NASController(model, crit, dev_list).to(device=dev)
    arch = build_arch_optim(config.arch_optim.type, config.arch_optim, model)
    # genotype
    if config.genotypes.disable_dag:
        model.to_genotype = model.to_genotype_ops
    if config.genotypes.use_slot:
        model.to_genotype_ops = model.to_genotype_slots
    # init
    model.init_model(config.init)
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.search,
        'chkpt_path': chkpt,
        'expman': expman,
        'train_loader': trn_loader,
        'valid_loader': val_loader,
        'model': model,
        'arch_optim': arch,
        'writer': writer,
        'logger': logger,
        'device': dev,
    }


def init_all_augment(config, name, exp_root_dir, chkpt, device, genotype, convert_fn=None):
    config = load_config(config, name, excludes=['search'])
    # dir
    expman = ExpManager(exp_root_dir)
    logger = utils.get_logger(expman.logs_path, name, config.log)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
    # device
    dev, dev_list = utils.init_device(config.device, device)
    # data
    val_loader = load_data(config.augment.data, validation=True)
    trn_loader = load_data(config.augment.data, validation=False)
    # primitives
    gt.set_primitives(config.primitives)
    # net
    Slot.reset()
    config.ops.affine = True
    configure_ops(config.ops)
    net = build_arch_space(config.model.type, config.model)
    # layers
    if not isinstance(convert_fn, list):
        convert_fn = [convert_fn]
    layer_convert_fn = convert_fn[:-1]
    layers_conf = config.get('layers', None)
    if not layers_conf is None:
        convert_from_layers(net, layers_conf, layer_convert_fn)
    # op
    drop_path = config.augment.drop_path_prob > 0.0
    op_convert_fn = convert_fn[-1]
    if genotype is None:
        if op_convert_fn is None and hasattr(net, 'get_predefined_augment_converter'):
            op_convert_fn = net.get_predefined_augment_converter()
        model = convert_from_predefined_net(net, op_convert_fn, drop_path)
    else:
        if op_convert_fn is None and hasattr(net, 'get_genotype_augment_converter'):
            op_convert_fn = net.get_genotype_augment_converter()
        genotype = gt.get_genotype(config.genotypes, genotype)
        model = convert_from_genotype(net, genotype, op_convert_fn, drop_path)
    # controller
    crit = utils.get_net_crit(config.criterion)
    model = NASController(model, crit, dev_list).to(device=dev)
    # init
    model.init_model(config.init)
    # chkpt
    chkpt = None if not chkpt is None and not os.path.isfile(chkpt) else chkpt
    return {
        'config': config.augment,
        'chkpt_path': chkpt,
        'expman': expman,
        'train_loader': trn_loader,
        'valid_loader': val_loader,
        'model': model,
        'writer': writer,
        'logger': logger,
        'device': dev,
    }