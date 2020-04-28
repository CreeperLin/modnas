# -*- coding: utf-8 -*-
# modified from https://github.com/HarryVolek/PyTorch_Speaker_Verification
import yaml
import copy
import logging
logger = logging.getLogger('config')

def load_config(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    config_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            config_dict[k] = v
    return config_dict


def parse_data_str(dstr):
    if dstr.isnumeric():
        return int(dstr)
    else:
        try:
            return float(dstr)
        except ValueError:
            return dstr


class Config(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, file=None, dct={}):
        super().__init__()
        if not file is None:
            dct = load_config(file)
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Config(None, value)
            elif isinstance(value, list):
                for i in range(len(value)):
                    if hasattr(value[i], 'keys'):
                        value[i] = Config(None, value[i])
            self[key] = value
        yaml.add_representer(Config, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

    def __deepcopy__(self, memo):
        return Config(None, copy.deepcopy(dict(self)))

    def __str__(self):
        return yaml.dump(dict(self), default_flow_style=False)

    @staticmethod
    def get_value(config, key):
        keywords = key.split('.')
        val = config[keywords[0]]
        if len(keywords) == 1: return val
        elif val is None: raise ValueError('invalid key: {}'.format(keywords[0]))
        return Config.get_value(val, '.'.join(keywords[1:]))

    @staticmethod
    def set_value(config, key, value):
        keywords = key.split('.')
        if len(keywords) == 1: 
            config[keywords[0]] = value
            return
        val = config.get(keywords[0], None)
        if val is None:
            raise ValueError('invalid key: {}'.format(keywords[0]))
        else:
            Config.set_value(val, '.'.join(keywords[1:]), value)

    @staticmethod
    def apply(config, dct):
        if isinstance(dct, dict):
            dct = Config(dct=dct)
        elif isinstance(dct, list):
            dct = {k: parse_data_str(v) for (k, v) in [item.split('=') for item in dct]}
        else:
            raise ValueError('unsupported apply type: {}'.format(type(dct)))
        for k, v in dct.items():
            Config.set_value(config, k, v)

    @staticmethod
    def merge(src, dest, overwrite=True):
        if isinstance(src, dict) and isinstance(dest, dict):
            for k, v in dest.items():
                if k not in src:
                    src[k] = v
                    logger.warning('merge_config: add key {}'.format(k))
                else:
                    src[k] = Config.merge(src[k], v, overwrite)
        elif isinstance(src, list) and isinstance(dest, list):
            logger.warning('merge_config: extend list: {} + {}'.format(src, dest))
            src.extend(dest)
        elif overwrite:
            src = dest
            logger.warning('merge_config: overwrite: {} -> {}'.format(src, dest))
        return src

    @staticmethod
    def load(conf):
        if isinstance(conf, Config):
            config = conf
        elif isinstance(conf, str):
            config = Config(file=conf)
        elif isinstance(conf, dict):
            config = Config(dct=conf)
        else:
            raise ValueError('invalid config type')
        return config
