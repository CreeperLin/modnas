from ..utils.exp_manager import ExpManager
from ..data_provider.dataloader import load_data
from ..arch_space.constructor import convert_from_predefined_net
from ..arch_space.constructor import convert_from_genotype
from ..core.ops import configure_ops
from ..arch_space import build_arch_space
from ..arch_space.constructor import Slot
from ..core.nas_modules import NASModule, build_nas_controller
from ..arch_optim import build_arch_optim
from .. import utils as utils
from ..arch_space import genotypes as gt

def init_all_search(config, name, exp_root_dir, device, genotype=None, convert_fn=None):
    # dir
    expman = ExpManager(exp_root_dir)
    logger = utils.get_logger(expman.logs_path, name)
    writer = utils.get_writer(expman.writer_path, config.log.writer)
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
    NASModule.reset()
    Slot.reset()
    configure_ops(config.ops)
    net = build_arch_space(config.model.type, config.model)
    mixed_op_args = config.mixed_op.get('args', {})
    drop_path = config.search.drop_path_prob > 0.0
    if genotype is None:
        supernet = convert_from_predefined_net(net, convert_fn, drop_path, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
    else:
        genotype = gt.get_genotype(None, genotype)
        supernet = convert_from_genotype(net, genotype, convert_fn, drop_path, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
    # model
    crit = utils.get_net_crit(config.criterion)
    model = build_nas_controller(supernet, crit, dev, dev_list)
    arch = build_arch_optim(config.arch_optim.type, config.arch_optim, model)
    return {
        'expman': expman,
        'train_loader': trn_loader,
        'valid_loader': val_loader,
        'model': model,
        'arch_optim': arch,
        'writer': writer,
        'logger': logger,
        'device': dev,
    }


def init_all_augment(config, name, exp_root_dir, device, genotype, convert_fn=None):
    # dir
    expman = ExpManager(exp_root_dir)
    logger = utils.get_logger(expman.logs_path, name)
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
    configure_ops(config.ops)
    net = build_arch_space(config.model.type, config.model)
    drop_path = config.augment.drop_path_prob > 0.0
    if genotype is None:
        if convert_fn is None and hasattr(net, 'get_default_converter'):
            convert_fn = net.get_default_converter()
        supernet = convert_from_predefined_net(net, convert_fn, drop_path)
    else:
        genotype = gt.get_genotype(None, genotype)
        supernet = convert_from_genotype(net, genotype, convert_fn, drop_path)
    # model
    crit = utils.get_net_crit(config.criterion)
    model = build_nas_controller(supernet, crit, dev, dev_list)
    return {
        'expman': expman,
        'train_loader': trn_loader,
        'valid_loader': val_loader,
        'model': model,
        'writer': writer,
        'logger': logger,
        'device': dev,
    }