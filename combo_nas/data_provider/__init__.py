import copy
from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('data_provider')
from . import dataloader
from . import dataset
from . import default


def get_data_provider(config, logger):
    data_conf = config.get('data', {})
    data_type = data_conf.get('type')
    data_args = data_conf.get('args', {})
    trn_data_conf = config.get('train_data', {})
    val_data_conf = config.get('valid_data', {})
    trn_data_args = copy.deepcopy(data_args)
    val_data_args = copy.deepcopy(data_args)
    trn_data_args.update(trn_data_conf.get('args', {}))
    val_data_args.update(val_data_conf.get('args', {}))
    dloader_conf = config.get('data_loader', None)
    data_prov_conf = config.get('data_provider', {})
    trn_data, val_data = None, None
    if trn_data_conf or data_type:
        trn_data = dataset.build(trn_data_conf.get('type', data_type), **trn_data_args)
    if val_data_conf:
        val_data = dataset.build(val_data_conf.get('type', data_type), **val_data_args)
    data_provd_args = data_prov_conf.get('args', {})
    if dloader_conf is not None:
        trn_loader, val_loader = dataloader.build(dloader_conf.type,
                                                  trn_data=trn_data,
                                                  val_data=val_data,
                                                  **dloader_conf.get('args', {}))
        data_provd_args['train_loader'] = trn_loader
        data_provd_args['valid_loader'] = val_loader
    elif not data_prov_conf:
        return None
    data_prov = build(data_prov_conf.get('type', 'Default'), logger=logger, **data_provd_args)
    return data_prov
