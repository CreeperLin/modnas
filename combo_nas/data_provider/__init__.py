# -*- coding: utf-8 -*-
from . import dataloader, torch_dataloader, torch_dataset

def load_data(config, validation):
    dloader_config = config.get('data_loader', None)
    if dloader_config is None:
        return None, None
    return dataloader.build(dloader_config.type, data_config=config.data, validation=validation,
                            **dloader_config.get('args',{}))
