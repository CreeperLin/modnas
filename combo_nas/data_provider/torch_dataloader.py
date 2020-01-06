import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from .prefetcher import fast_collate, data_prefetcher

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_torch_dataloader(config, metadata):
    dataset, root, mean, stddev, validation = metadata
    prefetch = config.prefetch
    size = config.get('size', 1)
    collate_fn = fast_collate if prefetch else None

    if dataset == 'cifar10':
        dset = datasets.CIFAR10
        trn_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        val_transf = []
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100
        trn_transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        val_transf = []
    elif dataset == 'mnist':
        dset = datasets.MNIST
        trn_transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
        val_transf = []
    elif dataset == 'fashionmnist':
        dset = datasets.FashionMNIST
        trn_transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
        val_transf = []
    elif dataset == 'imagenet':
        dset = datasets.ImageFolder
        trn_transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    elif dataset == 'image':
        dset = datasets.ImageFolder
        trn_transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    if config.jitter:
        trn_transf.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    cutout = config.cutout

    if not prefetch:
        normalize = [transforms.ToTensor(), transforms.Normalize(mean, stddev)]
        trn_transf.extend(normalize)
        val_transf.extend(normalize)
        if cutout > 0:
            trn_transf.append(Cutout(cutout))

    if dset == datasets.ImageFolder:
        if validation:
            data = dset(root, transform=transforms.Compose(val_transf))
        else:
            data = dset(root, transform=transforms.Compose(trn_transf))
    elif validation:
        data = dset(root, train=False,
                    transform=transforms.Compose(val_transf), download=True)
    else:
        data = dset(root, train=True,
                    transform=transforms.Compose(trn_transf), download=True)

    sp_ratio = config.split_ratio
    trn_loader = None
    val_loader = None
    extra_kwargs = {
        'num_workers': config.workers,
        'pin_memory': True,
        # 'collate_fn': collate_fn,
    }
    if not collate_fn is None:
        # backward compatibility for pytorch < 1.2.0
        extra_kwargs['collate_fn'] = collate_fn
    n_data = int(size * len(data))
    indices = list(range(n_data))
    if sp_ratio > 0 and sp_ratio < 1.0:
        split = int(n_data * sp_ratio)
        logging.info('data_provider: split data: {}/{}'.format(split, n_data-split))
        trn_sampler = SubsetRandomSampler(indices[:split])
        val_sampler = SubsetRandomSampler(indices[split:])
        trn_loader = DataLoader(data,
                                batch_size=config.trn_batch_size,
                                sampler=trn_sampler,
                                **extra_kwargs)
        val_loader = DataLoader(data,
                                batch_size=config.val_batch_size,
                                sampler=val_sampler,
                                **extra_kwargs)
    elif validation:
        val_sampler = SubsetRandomSampler(indices)
        val_loader = DataLoader(data,
                                batch_size=config.val_batch_size,
                                sampler=val_sampler, **extra_kwargs)
    else:
        trn_sampler = SubsetRandomSampler(indices)
        trn_loader = DataLoader(data,
                                batch_size=config.trn_batch_size,
                                sampler=trn_sampler, **extra_kwargs)

    if prefetch:
        if not trn_loader is None: trn_loader = data_prefetcher(trn_loader, mean, stddev, cutout)
        if not val_loader is None: val_loader = data_prefetcher(val_loader, mean, stddev, cutout)
    if trn_loader is None: return val_loader
    if val_loader is None: return trn_loader
    return trn_loader, val_loader
