import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .prefetcher import fast_collate, DataPrefetcher
from .dataloader import register_as
from .dataset import build, get_metadata

def train_valid_split(trn_idx, train_labels, valid_size, n_classes):
    random.shuffle(trn_idx)
    # return trn_idx[valid_size:], trn_idx[:valid_size]
    train_idx, valid_idx = [], []
    per_class_remain = [valid_size // n_classes] * n_classes
    for i in range(valid_size % n_classes):
        per_class_remain[i] += 1
    for idx in trn_idx:
        label = train_labels[idx]
        if isinstance(label, float):
            label = int(label)
        elif isinstance(label, np.ndarray):
            label = np.argmax(label)
        else:
            assert isinstance(label, int)
        if per_class_remain[label] > 0:
            valid_idx.append(idx)
            per_class_remain[label] -= 1
        else:
            train_idx.append(idx)
    return train_idx, valid_idx


@register_as('pytorch')
def get_torch_dataloader(data_config, validation,
                         trn_batch_size=64, val_batch_size=64,
                         workers=2, prefetch=False, collate_fn=None,
                         train_size=0, train_ratio=1., train_seed=1,
                         valid_size=0, valid_ratio=0., valid_seed=1):
    if prefetch:
        collate_fn = fast_collate
    data_args = data_config.get('args', {})
    if prefetch:
        data_args['to_tensor'] = False
        cutout = data_args['cutout']
        data_args['cutout'] = 0
        metadata = get_metadata(data_args['dataset'])
        mean, stddev = metadata['mean'], metadata['stddev']
    trn_data, val_data = build(data_config.type, validation=validation, **data_args)
    trn_loader = None
    val_loader = None
    extra_kwargs = {
        'num_workers': workers,
        'pin_memory': True,
        # 'collate_fn': collate_fn,
    }
    if not collate_fn is None:
        # backward compatibility for pytorch < 1.2.0
        extra_kwargs['collate_fn'] = collate_fn

    n_train_data = len(trn_data)
    n_valid_data = 0 if val_data is None else len(val_data)
    if train_size <= 0:
        train_size = int(n_train_data * min(train_ratio, 1.))
    if train_size < n_train_data:
        random.seed(train_seed)
        trn_idx = random.sample(range(n_train_data), train_size)
    else:
        trn_idx = list(range(n_train_data))
    if validation:
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(n_valid_data * min(valid_ratio, 1.))
        if valid_size < n_valid_data:
            random.seed(valid_seed)
            val_idx = random.sample(range(n_valid_data), valid_size)
        else:
            val_idx = list(range(n_valid_data))
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(val_data,
                                batch_size=val_batch_size,
                                sampler=val_sampler, **extra_kwargs)
        trn_sampler = SubsetRandomSampler(trn_idx)
        trn_loader = DataLoader(trn_data,
                                batch_size=trn_batch_size,
                                sampler=trn_sampler, **extra_kwargs)
        logging.info('data_loader: trn: {} val: {}'.format(
            len(trn_idx), len(val_idx)))
    else:
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(train_size * min(valid_ratio, 1.))
        if valid_size > 0:
            if hasattr(trn_data, 'targets'):
                labels = [c for c in trn_data.targets]
            elif hasattr(trn_data, 'samples'):
                labels = [c for _, c in trn_data.samples]
            elif hasattr(trn_data, 'train_labels'): # backward compatibility for pytorch<1.2.0
                labels = trn_data.train_labels
            else:
                raise RuntimeError('data labels not found')
            if hasattr(trn_data, 'classes'):
                n_classes = len(trn_data.classes)
            else:
                n_classes = len(set(labels))
            random.seed(valid_seed)
            trn_idx, val_idx = train_valid_split(trn_idx, labels, valid_size, n_classes)
            trn_sampler = SubsetRandomSampler(trn_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            trn_loader = DataLoader(trn_data,
                                    batch_size=trn_batch_size,
                                    sampler=trn_sampler,
                                    **extra_kwargs)
            val_loader = DataLoader(trn_data,
                                    batch_size=val_batch_size,
                                    sampler=val_sampler,
                                    **extra_kwargs)
            logging.info('data_loader: split: trn: {} val: {} cls: {}'.format(
                len(trn_idx), len(val_idx), n_classes))
        else:
            val_loader = None
            trn_sampler = SubsetRandomSampler(trn_idx)
            trn_loader = DataLoader(trn_data,
                                    batch_size=trn_batch_size,
                                    sampler=trn_sampler, **extra_kwargs)
            logging.info('data_loader: trn: {} no val'.format(len(trn_idx)))
    if prefetch:
        if not trn_loader is None: trn_loader = DataPrefetcher(trn_loader, mean, stddev, cutout)
        if not val_loader is None: val_loader = DataPrefetcher(val_loader, mean, stddev, cutout)
    return trn_loader, val_loader
