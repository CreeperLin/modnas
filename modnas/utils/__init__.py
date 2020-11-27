import os
import sys
import time
import logging
import numpy as np
import torch
from ..version import __version__
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def get_exp_name(config):
    if 'name' in config:
        return config['name']
    return time.strftime('%Y%m%d%H%M.', time.localtime()) + hex(hash(str(config))).split('x')[1][:4]


def merge_config(src, dest, overwrite=True):
    if isinstance(src, dict) and isinstance(dest, dict):
        for k, v in dest.items():
            if k not in src:
                src[k] = v
                logging.warning('merge_config: add key {}'.format(k))
            else:
                src[k] = merge_config(src[k], v, overwrite)
    elif isinstance(src, list) and isinstance(dest, list):
        logging.warning('merge_config: extend list: {} + {}'.format(src, dest))
        src.extend(dest)
    elif overwrite:
        logging.warning('merge_config: overwrite: {} -> {}'.format(src, dest))
        src = dest
    return src


def env_info():
    """Return dependency versions."""
    return 'environment info:\nmodnas: {}\npython: {}\npytorch: {}\ncudnn: {}'.format(
        __version__,
        sys.version.split()[0],
        torch.__version__,
        torch.backends.cudnn.version(),
    )


def parse_device(device):
    """Return device ids from config."""
    if not isinstance(device, str):
        return []
    device = device.lower()
    if device in ['cpu', 'nil', 'none']:
        return []
    if device == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in device.split(',')]


def check_config(config):
    """Check config and set default values."""
    def check_field(config, field, default):
        cur_key = ''
        idx = -1
        keys = field.split('.')
        cur_dict = config
        try:
            for idx in range(len(keys)):
                cur_key = keys[idx]
                if cur_key == '*':
                    if isinstance(cur_dict, list):
                        key_list = ['#{}'.format(i) for i in range(len(cur_dict))]
                    else:
                        key_list = cur_dict.keys()
                    for k in key_list:
                        keys[idx] = k
                        nfield = '.'.join(keys)
                        if check_field(config, nfield, default):
                            return True
                    return False
                if cur_key.startswith('#'):
                    cur_key = int(cur_key[1:])
                cur_dict = cur_dict[cur_key]
        except KeyError:
            if idx != len(keys) - 1:
                logging.warning('check_config: key \'{}\' in field \'{}\' missing'.format(cur_key, field))
            else:
                logging.warning('check_config: setting field \'{}\' to default: {}'.format(field, default))
                cur_dict[cur_key] = default
        return False

    defaults = {
        'estimator.*.save_arch_desc': True,
        'estimator.*.save_freq': 0,
        'estimator.*.arch_update_epoch_start': 0,
        'estimator.*.arch_update_epoch_intv': 1,
        'estimator.*.arch_update_intv': -1,
        'estimator.*.arch_update_batch': 1,
        'estimator.*.criterion': 'CrossEntropyLoss',
        'estimator.*.metrics': 'ValidateMetrics',
    }

    for key, val in defaults.items():
        check_field(config, key, val)


def init_device(device='all', seed=11235):
    """Initialize device and set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_ids = parse_device(device)
    if not len(device_ids):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.cuda.set_device(device_ids[0])
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    return device, device_ids


def get_logger(log_dir, name, debug=False):
    """Return a new logger."""
    level = logging.DEBUG if debug else logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = os.path.join(log_dir, '%s-%d.log' % (name, time.time()))
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging.getLogger(name)


class DummyWriter():
    """A no-op writer."""

    def __getattr__(self, item):
        """Return no-op."""
        def noop(*args, **kwargs):
            pass

        return noop


def get_writer(log_dir, enabled=False):
    """Return a new writer."""
    if enabled:
        if SummaryWriter is None:
            raise ValueError('module SummaryWriter is not found')
        writer = SummaryWriter(log_dir)
    else:
        writer = DummyWriter()
    return writer


def get_same_padding(kernel_size):
    """Return SAME padding size for convolutions."""
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def param_count(model, factor=0, divisor=1000):
    """Return number of model parameters."""
    return sum(p.data.nelement() for p in model.parameters()) / divisor**factor


def param_size(model, factor=0, divisor=1024):
    """Return size of model parameters."""
    return 4 * param_count(model) / divisor**factor


class AverageMeter():
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Compute the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


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


def format_time(sec):
    """Return formatted time in seconds."""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h %d m %d s" % (h, m, s)


class ETAMeter():
    """ETA Meter."""

    def __init__(self, total_steps, cur_steps=-1):
        self.total_steps = total_steps
        self.last_step = cur_steps
        self.last_time = time.time()
        self.speed = None

    def start(self):
        """Start timing."""
        self.last_time = time.time()

    def set_step(self, step):
        """Set current step."""
        self.speed = (step - self.last_step) / (time.time() - self.last_time + 1e-7)
        self.last_step = step
        self.last_time = time.time()

    def step(self, n=1):
        """Increment current step."""
        self.speed = n / (time.time() - self.last_time + 1e-7)
        self.last_step += n
        self.last_time = time.time()

    def eta(self):
        """Return ETA in seconds."""
        if self.speed is None:
            return 0
        return (self.total_steps - self.last_step) / (self.speed + 1e-7)

    def eta_fmt(self):
        """Return formatted ETA."""
        return format_time(self.eta())
