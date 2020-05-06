import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .prefetcher import fast_collate, DataPrefetcher
from .dataloader import register_as
from .dataset import build, get_metadata

def get_label_class(label):
    if isinstance(label, float):
        label_cls = int(label)
    elif isinstance(label, np.ndarray):
        label_cls = int(np.argmax(label))
    elif isinstance(label, int):
        label_cls = label
    else:
        raise ValueError('unsupported label type: {}'.format(label))
    return label_cls


def get_dataset_label(data):
    if hasattr(data, 'targets'):
        return [c for c in data.targets]
    if hasattr(data, 'samples'):
        return [c for _, c in data.samples]
    if hasattr(data, 'train_labels'): # backward compatibility for pytorch<1.2.0
        return data.train_labels
    if hasattr(data, 'test_labels'):
        return data.test_labels
    raise RuntimeError('data labels not found')


def filter_class(data_idx, labels, classes):
    return [idx for idx in data_idx if get_label_class(labels[idx]) in classes]


def train_valid_split(trn_idx, train_labels, class_size):
    random.shuffle(trn_idx)
    train_idx, valid_idx = [], []
    for idx in trn_idx:
        label_cls = get_label_class(train_labels[idx])
        if not label_cls in class_size:
            continue
        if class_size[label_cls] > 0:
            valid_idx.append(idx)
            class_size[label_cls] -= 1
        else:
            train_idx.append(idx)
    return train_idx, valid_idx


def map_data_label(data, mapping):
    labels = get_dataset_label(data)
    if hasattr(data, 'targets'):
        data.targets = [mapping.get(get_label_class(c), c) for c in labels]
    if hasattr(data, 'samples'):
        data.samples = [(s, mapping.get(get_label_class(c), c)) for s, c in data.samples]
    if hasattr(data, 'train_labels'):
        data.train_labels = [mapping.get(get_label_class(c), c) for c in labels]
    if hasattr(data, 'test_labels'):
        data.test_labels = [mapping.get(get_label_class(c), c) for c in labels]


@register_as('pytorch')
def get_torch_dataloader(data_config, validation, parallel_multiplier=1,
                         trn_batch_size=64, val_batch_size=64, classes=None,
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
    # classes
    trn_labels = get_dataset_label(trn_data)
    if hasattr(trn_data, 'classes'):
        class_name = trn_data.classes
    else:
        class_name = []
    all_classes = list(set(trn_labels))
    if isinstance(classes, list):
        all_classes = []
        for c in classes:
            if isinstance(c, str):
                idx = class_name.index(c)
                if idx == -1:
                    continue
                all_classes.append(idx)
            elif isinstance(c, int):
                all_classes.append(c)
            else:
                raise ValueError('invalid class type')
    elif isinstance(classes, int):
        random.seed(train_seed)
        all_classes = random.sample(all_classes, classes)
        logging.info('data_loader: random classes: {}'.format(all_classes))
    n_classes = len(all_classes)
    # index
    trn_idx = list(range(len(trn_data)))
    trn_idx = filter_class(trn_idx, trn_labels, all_classes)
    n_train_data = len(trn_idx)
    if train_size <= 0:
        train_size = int(n_train_data * min(train_ratio, 1.))
    if 0 < train_size < n_train_data:
        random.seed(train_seed)
        trn_idx = random.sample(trn_idx, train_size)
    if validation:
        val_labels = get_dataset_label(val_data)
        val_idx = list(range(len(val_data)))
        val_idx = filter_class(val_idx, val_labels, all_classes)
        n_valid_data = len(val_idx)
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(n_valid_data * min(valid_ratio, 1.))
        if 0 < valid_size < n_valid_data:
            random.seed(valid_seed)
            val_idx = random.sample(val_idx, valid_size)
    else:
        val_data = trn_data
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(train_size * min(valid_ratio, 1.))
        if valid_size > 0:
            random.seed(valid_seed)
            val_class_size = {}
            for i, c in enumerate(all_classes):
                val_class_size[c] = valid_size // n_classes + (1 if i < valid_size % n_classes else 0)
            trn_idx, val_idx = train_valid_split(trn_idx, trn_labels, val_class_size)
        else:
            val_idx = list()
    logging.info('data_loader: trn: {} val: {} cls: {}'.format(
                len(trn_idx), len(val_idx), n_classes))
    # map labels
    if not classes is None:
        mapping = {c:i for i, c in enumerate(all_classes)}
        map_data_label(trn_data, mapping)
        if not val_data is None:
            map_data_label(val_data, mapping)
    # dataloader
    trn_loader = val_loader = None
    trn_batch_size *= parallel_multiplier
    val_batch_size *= parallel_multiplier
    workers *= parallel_multiplier
    extra_kwargs = {
        'num_workers': workers,
        'pin_memory': True,
    }
    if not collate_fn is None:
        # backward compatibility for pytorch < 1.2.0
        extra_kwargs['collate_fn'] = collate_fn
    if len(trn_idx) > 0:
        trn_sampler = SubsetRandomSampler(trn_idx)
        trn_loader = DataLoader(trn_data,
                                batch_size=trn_batch_size,
                                sampler=trn_sampler,
                                **extra_kwargs)
    if len(val_idx) > 0:
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(val_data,
                                batch_size=val_batch_size,
                                sampler=val_sampler,
                                **extra_kwargs)
    if prefetch:
        if not trn_loader is None: trn_loader = DataPrefetcher(trn_loader, mean, stddev, cutout)
        if not val_loader is None: val_loader = DataPrefetcher(val_loader, mean, stddev, cutout)
    return trn_loader, val_loader
