# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from .BoT import *
from .config import Config
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    import adabound
except ImportError:
    adabound = None

def get_current_device():
    if not torch.cuda.is_available(): return 'cpu'
    return torch.cuda.current_device()

def parse_gpus(gpus):
    if gpus == 'cpu':
        return []
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def check_config(config, top_keys=[]):
    for k in top_keys:
        if not k in config:
            config[k] = Config()
    
    def check_field(config, field, default, required=False):
        cur_key = ''
        idx = -1
        keys = field.split('.')
        cur_dict = config
        try:
            for idx in range(len(keys)):
                cur_key = keys[idx]
                if cur_key == '*':
                    for k in cur_dict.keys():
                        keys[idx] = k
                        nfield = '.'.join(keys)
                        if check_field(config, nfield, default):
                            return True
                    return False
                cur_dict = cur_dict[cur_key]
        except KeyError:
            if idx != len(keys) - 1:
                logging.warning('check_config: key \'{}\' in field \'{}\' missing'.format(cur_key, field))
            elif required:
                logging.error('missing field \'{}\''.format(field))
                return True
            else:
                logging.warning('check_config: setting field \'{}\' to default: {}'.format(field, default))
                cur_dict[cur_key] = default
        return False

    flag = False

    defaults = {
        'data.type': 'pytorch',
        'data_loader.type': 'pytorch',
        'ops.ops_order': 'act_weight_bn',
        'ops.affine': False,
        'ops.bias': False,
        'log.writer': False,
        'log.debug': False,
        'device.seed': 2,
        'device.gpus': 'all',
        'genotypes.disable_dag': False,
        'genotypes.use_slot': True,
        'genotypes.gt_str': '',
        'genotypes.gt_path': '',
        'genotypes.to_args': {},
        'convert.search_args': {},
        'convert.augment_args': {},
        'estimator.*.save_gt': True,
        'estimator.*.save_freq': 0,
        'estimator.*.print_freq': 200,
        'estimator.*.drop_path_prob': 0.,
        'estimator.*.w_grad_clip': 0.,
        'estimator.*.aux_weight': 0.,
        'estimator.*.arch_update_epoch_start': 0,
        'estimator.*.arch_update_epoch_intv': 1,
        'estimator.*.arch_update_intv': -1,
        'estimator.*.arch_update_batch': 1,
    }

    for key, val in defaults.items():
        if check_field(config, key, val):
            flag = True
    if flag:
        raise ValueError('check_config: Failed')
    logging.info('check_config: OK')
    return False


def init_device(config, ovr_gpus):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if ovr_gpus is None:
        config.gpus = parse_gpus(config.gpus)
    else:
        config.gpus = parse_gpus(ovr_gpus)
    if len(config.gpus)==0:
        device = torch.device('cpu')
        return device, []
    device = torch.device("cuda")
    torch.cuda.set_device(config.gpus[0])

    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    logging.debug('device: {} {}'.format(device, config.gpus))
    return device, config.gpus


def get_logger(log_dir, name, config):
    level = logging.DEBUG if config.debug else logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path = os.path.join(log_dir, '%s-%d.log' % (name, time.time()))
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


class DummyWriter():
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


def get_writer(log_dir, enabled):
    if enabled:
        if SummaryWriter is None:
            raise ValueError('module SummaryWriter is not found')
        writer = SummaryWriter(log_dir)
    else:
        writer = DummyWriter()
    return writer

def get_net_crit(config):
    net_crit_args = config.get('args', {})
    crit_type = config.type
    if crit_type == 'LS':
        crit = CrossEntropyLossLS
    elif crit_type == 'CE':
        crit = nn.CrossEntropyLoss
    else:
        raise ValueError('Unsupported loss function: {}'.format(crit_type))
    return crit(**net_crit_args)

def get_optim(params, config):
    optim_args = config.get('args', {})
    if config.type == 'adam':
        optimizer = torch.optim.Adam
    elif config.type == 'adabound':
        if adabound is None:
            raise ValueError('module adabound not found')
        optimizer = adabound.AdaBound
    elif config.type == 'sgd':
        optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported: %s" % config.optimizer)
    return optimizer(params, **optim_args)

def get_lr_scheduler(optim, config, epochs):
    lr_type = config.type
    lr_args = config.get('args', {})
    if lr_type == 'cosine':
        if not 'T_max' in lr_args: lr_args['T_max'] = epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    elif lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR
    elif lr_type == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
    else:
        raise ValueError('unsupported lr scheduler: {}'.format(lr_type))
    return lr_scheduler(optim, **lr_args)

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(p.data.nelement() for p in model.parameters())
    return 4 * n_params / 1024. / 1024.

def param_count(model):
    """ Compute parameter count in million """
    n_params = sum(p.data.nelement() for p in model.parameters())
    return n_params / 1e6

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
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

def format_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h %d m %d s" % (h,m,s)

class ETAMeter():
    def __init__(self, tot_epochs, epoch, tot_step):
        self.tot_epochs = tot_epochs
        self.epoch = epoch
        self.tot_step = tot_step
        self.last_step = None
        self.last_time = None

    def start(self):
        self.last_step = -1
        self.last_time = time.time()

    def step(self, step):
        elps = time.time() - self.last_time
        eta = ((self.tot_epochs-self.epoch) * self.tot_step - step) * elps / (step-self.last_step)
        self.last_time = time.time()
        self.last_step = step
        return eta