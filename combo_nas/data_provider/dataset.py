from ..utils.registration import get_registry_utils
registry, register, get_builder, build, register_as = get_registry_utils('dataset')

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
