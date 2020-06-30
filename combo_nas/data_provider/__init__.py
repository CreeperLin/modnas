# -*- coding: utf-8 -*-
from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('data_provider')
from . import dataloader
from . import dataset
from .default import DefaultDataProvider

def load_data(config, device, validation, logger):
    data_prov_config = config.get('data_provider', None)
    if data_prov_config is None:
        dl_config = config.get('data_loader', None)
        if dl_config is None:
            data_prov = None
        else:
            trn_loader, val_loader = dataloader.build(dl_config.type,
                            data_config=config.data, validation=validation, **dl_config.get('args', {}))
            data_prov = DefaultDataProvider(trn_loader, val_loader, device=device, logger=logger)
    else:
        data_prov = build(data_prov_config.type, device=device, logger=logger, **data_prov_config.get('args', {}))
    return data_prov
