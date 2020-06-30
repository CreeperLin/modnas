import os
import numpy as np
import torch
from torchvision import transforms, datasets
from . import register_as

def get_metadata(dataset):
    if dataset == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        stddev = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        stddev = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif dataset == 'mnist':
        mean = [0.13066051707548254]
        stddev = [0.30810780244715075]
    elif dataset == 'fashionmnist':
        mean = [0.28604063146254594]
        stddev = [0.35302426207299326]
    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        stddev = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        stddev = [0, 0, 0]
    return {
        'mean': mean,
        'stddev': stddev,
    }


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


@register_as('ImageCls')
def get_torch_dataset(dataset, train_root, valid_root, validation, mean=None, stddev=None,
                      cutout=0, jitter=False, resize_scale=0.08, to_tensor=True):
    dataset = dataset.lower()
    meta = get_metadata(dataset)
    mean = meta['mean'] if mean is None else mean
    stddev = meta['stddev'] if stddev is None else stddev
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(valid_root, exist_ok=True)
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
            transforms.RandomResizedCrop(224, scale=(resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    elif dataset == 'image':
        dset = datasets.ImageFolder
        trn_transf = [
            transforms.RandomResizedCrop(224, scale=(resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        val_transf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError('unsupported dataset: {}'.format(dataset))

    if jitter is True or jitter == 'strong':
        trn_transf.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    elif jitter == 'normal':
        trn_transf.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
    if to_tensor:
        normalize = [transforms.ToTensor(), transforms.Normalize(mean, stddev)]
        trn_transf.extend(normalize)
        val_transf.extend(normalize)
    if cutout > 0:
        trn_transf.append(Cutout(cutout))

    trn_data = val_data = None
    if dset == datasets.ImageFolder:
        if validation:
            val_data = dset(valid_root, transform=transforms.Compose(val_transf))
        trn_data = dset(train_root, transform=transforms.Compose(trn_transf))
    else:
        if validation:
            val_data = dset(valid_root, train=False,
                            transform=transforms.Compose(val_transf), download=True)
        trn_data = dset(train_root, train=True,
                        transform=transforms.Compose(trn_transf), download=True)
    return trn_data, val_data
