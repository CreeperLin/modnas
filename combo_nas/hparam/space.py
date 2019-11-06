import numpy as np
import json
import copy

def named_hparams(config, prefix=''):
    for k, v in config.items():
        if isinstance(v, HParam):
            yield (prefix + k), v
        if isinstance(v, Config):
            for n, hp in named_hparams(config, prefix):
                yield (prefix+'.'+k+'.'+n), hp

class HParamSpace():
    def __init__(self, dct=None):
        self._length = None
        self.hp_map = dict() if dct is None else dct
        self.__getitem__ = self.hp_map.__getitem__
        self.__iter__ = self.hp_map.__iter__
        
    def add_hparam(self, name, hp):
        print(name, hp.val_range)
        self.hp_map[name] = hp

    def __len__(self):
        if self._length is None:
            self._length = int(np.prod([len(x) for x in self.hp_map.values()]))
        return self._length

    def get(self, index):
        hparams = dict()
        t = index
        for name, hp in self.hp_map.items():
            hparams[name] = hp.get(t % len(hp))
            t //= len(hp)
        return hparams

    def sample(self):
        return self.get(np.random.randint(len(self.space)))
    
    @staticmethod
    def build_from_config(config):
        space = HParamSpace()
        for n, hp in named_hparams(config):
            space.add_hparam(n, hp)
        print('hparam space', len(space))
        return space
        
    @staticmethod
    def build_from_json(path):
        space = HParamSpace()
        hp_json = json.load(open(path, 'r'))
        for k, v in hp_json.items():
            if isinstance(v, list):
                hp = HParam(range=v)
                space.add_hparam(k, hp)
            elif not k=='//':
                raise ValueError('support hparam in list format only')
        print('hparam space', len(space))
        return space
    
    def to_json(self, path):
        json.dump(self.hp_map, path)


class HParam():
    def __init__(self, range):
        self.val_range = range
        self._length = None
    
    def sample(self):
        return self.val_range[np.random.randint(len(self))]

    def get(self, index):
        return self.val_range[index]
    
    def get_index(self, value):
        return self.val_range.index(value)
    
    def __len__(self):
        if self._length is None:
            self._length = len(self.val_range)
        return self._length
    
    @staticmethod
    def from_string(hp_str):
        if isinstance(hp_str, str) and hp_str.startswith('__HParam'):
            return eval(hp_str[2:])
        else: return None


def build_hparam_space(path):
    return HParamSpace.build_from_json(path)