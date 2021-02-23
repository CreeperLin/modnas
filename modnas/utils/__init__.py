import os
import sys
import time
import logging
import importlib
import numpy as np
import hashlib
from ..version import __version__
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def import_modules(modules):
    """Import modules by name."""
    if modules is None:
        return
    if isinstance(modules, str):
        modules = [modules]
    for m in modules:
        if isinstance(m, str):
            importlib.import_module(m)
        elif isinstance(m, (list, tuple)):
            n, p = m
            import_file(n, p)
        else:
            raise ValueError('Invalid import spec')


def get_exp_name(config):
    if 'name' in config:
        return config['name']
    return '{}.{}'.format(hashlib.sha1(str(config).encode()).hexdigest()[:4], time.strftime('%Y%m%d%H%M', time.localtime()))


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
    """Return environment info."""
    info = {
        'platform': sys.platform,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'modnas': __version__,
    }
    return 'env info: {}'.format(', '.join(['{k}={{{k}}}'.format(k=k) for k in info])).format(**info)


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
        'estimator.*.metrics': 'ValidateMetrics',
    }

    for key, val in defaults.items():
        check_field(config, key, val)


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


def format_time(sec):
    """Return formatted time in seconds."""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h %d m %d s" % (h, m, s)


def format_value(value, binary=False, div=None, factor=None, prec=2, unit=True):
    units = [None, 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    div = (1024. if binary else 1000.) if div is None else div
    if factor is None:
        factor = 0
        tot_div = 1
        while value > tot_div:
            factor += 1
            tot_div *= div
    else:
        tot_div = div ** factor
    value = round(value // tot_div, prec)
    return '{} {}'.format(value, units[factor]) if unit else value


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
