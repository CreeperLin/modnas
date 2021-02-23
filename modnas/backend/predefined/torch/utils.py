import numpy as np
import torch
from modnas.utils import format_value


def init_device(device=None, seed=11235):
    """Initialize device and set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_dev_mem_used():
    return torch.cuda.memory_allocated() / 1024. / 1024.


def param_count(model, *args, format=True, **kwargs):
    """Return number of model parameters."""
    val = sum(p.data.nelement() for p in model.parameters())
    return format_value(val, *args, **kwargs) if format else val


def param_size(model, *args, **kwargs):
    """Return size of model parameters."""
    val = 4 * param_count(model)
    return format_value(val, *args, binary=True, **kwargs) if format else val


def model_summary(model):
    info = {
        'params': param_count(model, factor=2),
        'size': param_size(model, factor=2),
    }
    return 'Model summary: {}'.format(' ,'.join(['{k}={{{k}}}'.format(k=k) for k in info])).format(**info)


def clear_bn_running_statistics(model):
    """Clear BatchNorm running statistics."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()


def recompute_bn_running_statistics(model, trainer, num_batch=100, clear=True):
    """Recompute BatchNorm running statistics."""
    if clear:
        clear_bn_running_statistics(model)
    is_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(num_batch):
            try:
                trn_X, _ = trainer.get_next_train_batch()
            except StopIteration:
                break
            model(trn_X)
            del trn_X
    if not is_training:
        model.eval()
