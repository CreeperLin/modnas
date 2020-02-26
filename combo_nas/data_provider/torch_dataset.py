import os
import numpy as np
import torch
from torchvision import transforms, datasets
from .dataset import register_as, get_metadata

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


@register_as('pytorch')
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
