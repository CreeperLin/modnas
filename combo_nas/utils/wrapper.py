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
    trn_loader, val_loader = load_data(config.search.data, validation=False)
    # primitives
    gt.set_primitives(config.primitives)
    # net
    NASModule.reset()
    Slot.reset()
    configure_ops(config.ops)
    net = build_arch_space(config.model.type, config.model)
    mixed_op_args = config.mixed_op.get('args', {})
    if genotype is None:
        supernet = convert_from_predefined_net(net, convert_fn, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
    else:
        genotype = gt.get_genotype(None, genotype)
        supernet = convert_from_genotype(net, genotype, convert_fn, mixed_op_cls=config.mixed_op.type, **mixed_op_args)
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
    if genotype is None:
        supernet = convert_from_predefined_net(net, convert_fn)
    else:
        genotype = gt.get_genotype(None, genotype)
        supernet = convert_from_genotype(net, genotype, convert_fn)
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