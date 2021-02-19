from ..registry.data_provider import register, get_builder, build, register_as
from ..utils import merge_config
from . import dataloader
from . import dataset
from . import default


def get_data(configs):
    config = None
    for conf in configs:
        if conf is None:
            continue
        config = conf if config is None else merge_config(config, conf)
    if config is None:
        return None
    return dataset.build(config)


def get_data_provider(config, logger):
    """Return a new DataProvider."""
    trn_data = get_data([config.get('data'), config.get('train_data')])
    val_data = get_data([config.get('data'), config.get('valid_data')])
    dloader_conf = config.get('data_loader', None)
    data_prov_conf = config.get('data_provider', {})
    data_provd_args = data_prov_conf.get('args', {})
    if dloader_conf is not None:
        trn_loader, val_loader = dataloader.build(dloader_conf,
                                                  trn_data=trn_data,
                                                  val_data=val_data)
        data_provd_args['train_loader'] = trn_loader
        data_provd_args['valid_loader'] = val_loader
    elif not data_prov_conf:
        return None
    data_prov = build(data_prov_conf.get('type', 'DefaultDataProvider'), logger=logger, **data_provd_args)
    return data_prov
