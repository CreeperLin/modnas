""" Architect controls architecture of cell by computing gradients of alphas """
import math
import copy
import torch
from ..base import GradientBasedOptim
from ...utils import accuracy
from ...core.param_space import ArchParamSpace
from ...arch_space.mixed_ops import MixedOp

class DARTSOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, w_momentum, w_weight_decay, logger=None):
        super().__init__(space, a_optim, logger)
        self.v_net = None
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, lr, optimizer, estim):
        # forward & calc loss
        model = estim.model
        loss = estim.loss(trn_X, trn_y, mode='train') # L_trn(w)
        # compute gradient
        gradients = torch.autograd.grad(loss, model.parameters())
        # do virtual step (update gradient)
        with torch.no_grad():
            for w, vw, g in zip(model.parameters(), self.v_net.parameters(), gradients):
                m = optimizer.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - lr * (m + g + self.w_weight_decay*w))
            # synchronize alphas
            # (no need, same reference to alphas)
            # for a, va in zip(model.alphas(), self.v_net.alphas()):
            #     va.copy_(a)

    def step(self, estim):
        self.optim_reset()
        trn_X, trn_y = estim.get_cur_train_batch()
        val_X, val_y = estim.get_next_valid_batch()
        lr = estim.trainer.get_lr()
        optimizer = estim.trainer.get_optimizer()
        model = estim.model
        if self.v_net is None: self.v_net = copy.deepcopy(model)
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, lr, optimizer, estim)
        # calc unrolled loss
        loss = estim.loss(val_X, val_y, model=self.v_net, mode='valid') # L_val(w`)
        # compute gradient
        alphas = ArchParamSpace.tensor_values()
        v_alphas = tuple(alphas)
        v_weights = tuple(self.v_net.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        hessian = self.compute_hessian(dw, trn_X, trn_y, estim)
        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for a, da, h in zip(alphas, dalpha, hessian):
                a.grad = da - lr*h
        self.optim_step()

    def compute_hessian(self, dw, trn_X, trn_y, estim):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        model = estim.model
        alphas = ArchParamSpace.tensor_values()
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p += eps * d
        loss = estim.loss(trn_X, trn_y, mode='train')
        dalpha_pos = torch.autograd.grad(loss, alphas) # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p -= 2. * eps * d
        loss = estim.loss(trn_X, trn_y, mode='train')
        dalpha_neg = torch.autograd.grad(loss, alphas) # dalpha { L_trn(w-) }
        # recover w
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p += eps * d
        hessian = [(p-n) / 2.*eps.item() for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


class BinaryGateOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, n_samples, renorm, logger=None):
        super().__init__(space, a_optim, logger)
        self.n_samples = n_samples
        self.sample = (self.n_samples!=0)
        self.renorm = renorm and self.sample

    def step(self, estim):
        self.optim_reset()
        model = estim.model
        val_X, val_y = estim.get_next_valid_batch()
        # sample k
        if self.sample:
            for m in MixedOp.gen(model):
                m.sample_ops(n_samples=self.n_samples)
        # loss
        for m in MixedOp.gen(model):
            m.arch_param_grad(enabled=True)
        loss = estim.loss(val_X, val_y, mode='valid')
        # backward
        loss.backward()
        # renormalization
        if not self.renorm:
            self.optim_step()
        else:
            with torch.no_grad():
                prev_pw = []
                for p, m in self.space.tensor_param_modules():
                    s_op = m.s_op
                    pdt = p.detach()
                    pp = pdt.index_select(-1, torch.tensor(s_op).to(p.device))
                    if pp.size() == pdt.size(): continue
                    k = torch.sum(torch.exp(pdt)) / torch.sum(torch.exp(pp)) - 1
                    prev_pw.append(k)

            self.optim_step()

            with torch.no_grad():
                for kprev, (p, m) in zip(prev_pw, self.space.tensor_param_modules()):
                    s_op = m.s_op
                    pdt = p.detach()
                    pp = pdt.index_select(-1, torch.tensor(s_op).to(p.device))
                    k = torch.sum(torch.exp(pdt)) / torch.sum(torch.exp(pp)) - 1
                    for i in s_op:
                        p[i] += (torch.log(k) - torch.log(kprev))

        for m in MixedOp.gen(model):
            m.arch_param_grad(enabled=False)
            m.reset_ops()


class DirectGradOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, logger=None):
        super().__init__(space, a_optim, logger)

    def step(self, estim):
        self.optim_step()
        self.optim_reset()


class DirectGradBiLevelOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, logger=None):
        super().__init__(space, a_optim, logger)

    def step(self, estim):
        self.optim_reset()
        val_X, val_y = estim.get_next_valid_batch()
        loss = estim.loss(val_X, val_y, mode='valid')
        loss.backward()
        self.optim_step()


class REINFORCEOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, batch_size, logger=None):
        super().__init__(space, a_optim, logger)
        self.batch_size = batch_size
        self.baseline = None
        self.baseline_decay_weight = 0.99

    def step(self, estim):
        model = estim.model
        self.optim_reset()
        grad_batch = []
        reward_batch = []
        val_X, val_y = estim.get_next_valid_batch()
        for _ in range(self.batch_size):
            acc_top1, _ = accuracy(estim.model(val_X), val_y, topk=(1, 5))
            # calculate reward according to net_info
            # reward = estim.get_score(estim.compute_metrics()) + acc_top1.item()
            reward = acc_top1.item()
            # loss term
            obj_term = 0
            for m in MixedOp.gen(model):
                p = m.alpha()
                if p.grad is not None:
                    p.grad.data.zero_()
                path_prob = m.prob()
                smpl = m.s_path_f
                path_prob_f = path_prob.index_select(-1, torch.tensor(smpl).to(path_prob.device))
                obj_term = obj_term + torch.log(path_prob_f)
            loss = -obj_term
            # backward
            loss.backward()
            # take out gradient dict
            grad_list = []
            for m in MixedOp.gen(model):
                p = m.alpha()
                grad_list.append(p.grad.data.clone())
            grad_batch.append(grad_list)
            reward_batch.append(reward)

        # update baseline function
        avg_reward = sum(reward_batch) / self.batch_size
        if self.baseline is None:
            self.baseline = avg_reward
        else:
            self.baseline += self.baseline_decay_weight * (avg_reward - self.baseline)
        # assign gradients
        for idx, m in enumerate(MixedOp.gen(model)):
            p = m.alpha()
            p.grad.data.zero_()
            for j in range(self.batch_size):
                p.grad.data += (reward_batch[j] - self.baseline) * grad_batch[j][idx]
            p.grad.data /= self.batch_size
        # apply gradients
        self.optim_step()


class GumbelAnnealingOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, init_temp=1e4, exp_anneal_rate=0.0015,
                 anneal_interval=1, restart_period=None, logger=None):
        super().__init__(space, a_optim, logger)
        self.init_temp = init_temp
        self.exp_anneal_rate = exp_anneal_rate
        self.temp = self.init_temp
        if restart_period is None:
            restart_period = 0
        self.restart_period = int(restart_period)
        self.anneal_interval = anneal_interval
        self.cur_step = 0

    def step(self, estim):
        self.optim_reset()
        model = estim.model
        self.apply_temp(model)
        val_X, val_y = estim.get_next_valid_batch()
        loss = estim.loss(val_X, val_y, mode='valid')
        loss.backward()
        self.optim_step()
        self.cur_step += 1
        if self.restart_period > 0 and self.cur_step >= self.restart_period:
            self.cur_step = 0
        intv = self.anneal_interval
        if self.cur_step % intv == 0:
            self.temp = self.init_temp * math.exp(-self.exp_anneal_rate * self.cur_step / intv)

    def apply_temp(self, model):
        for m in MixedOp.gen(model):
            m.set_temperature(self.temp)
