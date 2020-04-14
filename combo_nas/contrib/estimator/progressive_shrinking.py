import itertools
import time
import random
import torch
import torch.nn as nn
from combo_nas.estimator.base import EstimatorBase
from combo_nas.estimator import register_as
from combo_nas.contrib.arch_space.elastic.spatial import ElasticSpatial
from combo_nas.contrib.arch_space.elastic.sequential import ElasticSequential
from combo_nas import utils

@register_as('ProgressiveShrinking')
class ProgressiveShrinkingEstimator(EstimatorBase):
    def __init__(self, *args, stages, subnet_batch_size=1, use_ratio=False,
                 stage_rerank_spatial=True,
                 save_stage=False, reset_stage_training=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.stages = stages
        self.subnet_batch_size = subnet_batch_size
        self.use_ratio = use_ratio
        self.save_stage = save_stage
        self.reset_stage_training = reset_stage_training
        self.spatial_candidates = None
        self.sequential_candidates = None
        self.subnet_results = dict()
        self.cur_stage = -1
        self.stage_rerank_spatial = stage_rerank_spatial
        self.print_model_info()

    def reset_training(self):
        self.init_epoch = -1
        self.w_optim = utils.get_optimizer(self.model.weights(), self.config.w_optim)
        self.lr_scheduler = utils.get_lr_scheduler(self.w_optim, self.config.lr_scheduler,
                                                   self.config.epochs)

    def set_candidates(self, candidates):
        sp_cands = candidates.get('spatial', None)
        sq_cands = candidates.get('sequential', None)
        self.set_spatial_candidates(sp_cands)
        self.set_sequential_candidates(sq_cands)

    def set_sequential_candidates(self, candidates):
        n_groups = ElasticSequential.num_groups()
        if n_groups == 0 or candidates is None:
            candidates = [[None]]
        elif not isinstance(candidates[0], (list, tuple)):
            candidates = [candidates] * n_groups
        self.sequential_candidates = candidates
        self.logger.info('set sequential candidates: {}'.format(self.sequential_candidates))

    def set_spatial_candidates(self, candidates):
        n_groups = ElasticSpatial.num_groups()
        if n_groups == 0 or candidates is None:
            candidates = [[None]]
        elif not isinstance(candidates[0], (list, tuple)):
            candidates = [candidates] * n_groups
        self.spatial_candidates = candidates
        self.logger.info('set spatial candidates: {}'.format(self.spatial_candidates))

    def randomize(self, seed=None):
        if seed is None:
            seed = time.time()
        random.seed(seed)

    def apply_subnet_config(self, config):
        self.logger.debug('set subnet: {}'.format(config))
        spatial_config = config.get('spatial', None)
        if not spatial_config is None:
            for width, sp_g in zip(spatial_config, ElasticSpatial.groups()):
                if self.use_ratio:
                    sp_g.set_width_ratio(width)
                else:
                    sp_g.set_width(width)
        sequential_config = config.get('sequential', None)
        if not sequential_config is None:
            for depth, sp_g in zip(sequential_config, ElasticSequential.groups()):
                if self.use_ratio:
                    sp_g.set_depth_ratio(depth)
                else:
                    sp_g.set_depth(depth)

    def sample_spatial_config(self, seed=None):
        self.randomize(seed)
        spatial_config = []
        for i in range(ElasticSpatial.num_groups()):
            width = random.choice(self.spatial_candidates[i])
            spatial_config.append(width)
        return {
            'spatial': spatial_config,
        }

    def sample_sequential_config(self, seed=None):
        self.randomize(seed)
        sequential_config = []
        for i in range(ElasticSequential.num_groups()):
            depth = random.choice(self.sequential_candidates[i])
            sequential_config.append(depth)
        return {
            'sequential': sequential_config,
        }

    def sample_config(self, seed=None):
        config = dict()
        if not self.spatial_candidates is None:
            config.update(self.sample_spatial_config(seed=seed))
        if not self.sequential_candidates is None:
            config.update(self.sample_sequential_config(seed=seed))
        return config

    def loss_logits(self, X, y, model=None, mode=None):
        model = self.model if model is None else model
        if mode == 'train':
            subnet_logits = []
            num_subnets = self.subnet_batch_size
            visited = set()
            loss = None
            for _ in range(num_subnets):
                config = self.sample_config(seed=None)
                key = str(config)
                if key in visited:
                    continue
                if not loss is None:
                    loss.backward()
                self.apply_subnet_config(config)
                logits = model.logits(X)
                loss = self.criterion(X, logits, y, mode=mode)
                subnet_logits.append(logits)
                visited.add(key)
            if num_subnets > 1:
                logits = torch.mean(torch.stack(subnet_logits), dim=0)
        else:
            logits = model.logits(X)
            loss = self.criterion(X, logits, y, mode=mode)
        return loss, logits

    def train_stage(self):
        config = self.config
        tot_epochs = config.epochs
        if self.reset_stage_training:
            self.reset_training()
        for epoch in itertools.count(self.init_epoch+1):
            if epoch == tot_epochs: break
            # train
            # self.train_epoch(epoch, tot_epochs)
            # validate subnets
            results = self.validate_subnet(epoch, tot_epochs)
            for name, res in results.items():
                self.logger.info('Subnet {}: {}'.format(name, res))
            self.update_results(results)
            # save
            if config.save_freq != 0 and epoch % config.save_freq == 0:
                self.save_checkpoint(epoch)

    def update_results(self, results):
        for k, v in results.items():
            val = self.subnet_results.get(k, 0)
            self.subnet_results[k] = max(val, v)

    def state_dict(self):
        return {
            'cur_stage': self.cur_stage
        }

    def load_state_dict(self, state_dict):
        pass
        # if 'cur_stage' in state_dict:
        #     self.cur_stage = state_dict['cur_stage']

    def train(self):
        self.reset_training()
        for self.cur_stage in itertools.count(self.cur_stage+1):
            if self.cur_stage >= len(self.stages):
                break
            self.logger.info('running stage {}'.format(self.cur_stage))
            stage = self.stages[self.cur_stage]
            self.set_candidates(stage)
            if self.stage_rerank_spatial:
                pass
                self.rerank_spatial()
            self.train_stage()
            if self.save_stage:
                self.save_checkpoint(-1, 'stage_{}'.format(self.cur_stage))
        results = {
            'best_top1': max([acc for acc in self.subnet_results.values()]),
            'subnet_best_top1': self.subnet_results,
        }
        return results

    def search(self, optim):
        return self.train()

    def rerank_spatial(self):
        for g in ElasticSpatial.groups():
            g.set_spatial_rank()

    def validate_subnet(self, *args, configs=None, **kwargs):
        if configs is None:
            configs = dict()
            sp_len = ElasticSpatial.num_groups()
            sp_cand = self.spatial_candidates[0]
            sp_dim = len(sp_cand)
            sq_len = ElasticSequential.num_groups()
            sq_cand = self.sequential_candidates[0]
            sq_dim = len(sq_cand)
            for sp_idx, sq_idx in itertools.product(range(sp_dim), range(sq_dim)):
                sp_val = sp_cand[sp_idx]
                sq_val = sq_cand[sq_idx]
                sp_config = [sp_val] * sp_len
                sq_config = [sq_val] * sq_len
                conf = {
                    'spatial': sp_config,
                    'sequential': sq_config,
                }
                name = 'sp_{}_sq_{}'.format(sp_val, sq_val)
                configs[name] = conf
        results = dict()
        for name, conf in configs.items():
            self.logger.info('valid subnet: {}'.format(conf))
            self.apply_subnet_config(conf)
            self.recompute_bn_running_statistics(clear=False)
            val_top1 = self.validate_epoch(*args, **kwargs)
            results[name] = val_top1
        return results
