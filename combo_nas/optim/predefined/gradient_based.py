""" Architect controls architecture of cell by computing gradients of alphas """
import math
import copy
import torch
from ..base import GradientBasedOptim
from ...utils import accuracy

class DARTSOptim(GradientBasedOptim):
    """ Compute gradients of alphas """
    def __init__(self, space, a_optim, w_momentum, w_weight_decay, logger=None):
        super().__init__(space, a_optim, logger)
        self.v_net = None
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, lr, w_optim, estim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            lr: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        model = estim.model
        loss = estim.loss(trn_X, trn_y, mode='train') # L_trn(w)
        # compute gradient
        gradients = torch.autograd.grad(loss, model.weights())
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(model.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - lr * (m + g + self.w_weight_decay*w))
            # synchronize alphas
            # (no need, same reference to alphas)
            # for a, va in zip(model.alphas(), self.v_net.alphas()):
            #     va.copy_(a)

    def step(self, estim):
        """ Compute unrolled loss and backward its gradients
        Args:
            lr: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        self.optim_reset()
        trn_X, trn_y = estim.get_cur_trn_batch()
        val_X, val_y = estim.get_next_val_batch()
        lr = estim.get_lr()
        w_optim = estim.w_optim
        model = estim.model
        if self.v_net is None: self.v_net = copy.deepcopy(model)
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, lr, w_optim, estim)
        # calc unrolled loss
        loss = estim.loss(val_X, val_y, model=self.v_net, mode='valid') # L_val(w`)
        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        hessian = self.compute_hessian(dw, trn_X, trn_y, estim)
        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for alpha, da, h in zip(model.alphas(), dalpha, hessian):
                alpha.grad = da - lr*h
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
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(model.weights(), dw):
                p += eps * d
        loss = estim.loss(trn_X, trn_y, mode='train')
        dalpha_pos = torch.autograd.grad(loss, model.alphas()) # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(model.weights(), dw):
                p -= 2. * eps * d
        loss = estim.loss(trn_X, trn_y, mode='train')
        dalpha_neg = torch.autograd.grad(loss, model.alphas()) # dalpha { L_trn(w-) }
        # recover w
        with torch.no_grad():
            for p, d in zip(model.weights(), dw):
                p += eps * d
        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


class BinaryGateOptim(GradientBasedOptim):
    """ Compute gradients of alphas """
    def __init__(self, space, a_optim, n_samples, renorm, logger=None):
        super().__init__(space, a_optim, logger)
        self.n_samples = n_samples
        self.sample = (self.n_samples!=0)
        self.renorm = renorm and self.sample

    def step(self, estim):
        """ Compute unrolled loss and backward its gradients
        Args:
            lr: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        self.optim_reset()
        model = estim.model
        val_X, val_y = estim.get_next_val_batch()
        # sample k
        if self.sample:
            for m in model.mixed_ops():
                m.sample_ops(n_samples=self.n_samples)
        # loss
        for m in model.mixed_ops():
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

        for m in model.mixed_ops():
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
        val_X, val_y = estim.get_next_val_batch()
        loss = estim.loss(val_X, val_y, mode='valid')
        loss.backward()
        self.optim_step()


class REINFORCEOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, batch_size, logger=None):
        super().__init__(space, a_optim, logger)
        self.batch_size = batch_size
        self.baseline = None
        self.baseline_decay_weight = 0.99

    def reward(self, net_info):
        acc1 = net_info['acc1']
        acc_reward = acc1
        total_reward = acc_reward
        return total_reward

    def step(self, estim):
        self.optim_reset()
        grad_batch = []
        reward_batch = []
        net_info_batch = []
        val_X, val_y = estim.get_next_val_batch()
        for _ in range(self.batch_size):
            logits = estim.model.logits(val_X)
            acc1, _ = accuracy(logits, val_y, topk=(1, 5))
            net_info = {'acc1':acc1.item(), }
            net_info.update(estim.compute_metrics())
            net_info_batch.append(net_info)
            # calculate reward according to net_info
            reward = self.reward(net_info)
            # loss term
            obj_term = 0
            for m in estim.model.mixed_ops():
                p = m.arch_param_value('p')
                if p.grad is not None:
                    p.grad.data.zero_()
                path_prob = m.w_path_f
                smpl = m.s_path_f
                path_prob_f = path_prob.index_select(-1, torch.tensor(smpl).to(path_prob.device))
                obj_term = obj_term + torch.log(path_prob_f)
            loss = -obj_term
            # backward
            loss.backward()
            # take out gradient dict
            grad_list = []
            for m in estim.model.mixed_ops():
                p = m.arch_param_value('p')
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
        for idx, m in enumerate(estim.model.mixed_ops()):
            p = m.arch_param_value('p')
            p.grad.data.zero_()
            for j in range(self.batch_size):
                p.grad.data += (reward_batch[j] - self.baseline) * grad_batch[j][idx]
            p.grad.data /= self.batch_size
        # apply gradients
        self.optim_step()


class GumbelAnnealingOptim(GradientBasedOptim):
    def __init__(self, space, a_optim, init_temp=1e4, exp_anneal_rate=0.0015,
                 restart_period=None, logger=None):
        super().__init__(space, a_optim, logger)
        self.init_temp = init_temp
        self.exp_anneal_rate = exp_anneal_rate
        self.temp = self.init_temp
        if restart_period is None:
            restart_period = 0
        self.restart_period = int(restart_period)
        self.cur_step = 0

    def step(self, estim):
        self.optim_reset()
        model = estim.model
        self.apply_temp(model)
        val_X, val_y = estim.get_next_val_batch()
        loss = estim.loss(val_X, val_y, mode='valid')
        loss.backward()
        self.optim_step()
        self.cur_step += 1
        if self.restart_period > 0 and self.cur_step >= self.restart_period:
            self.cur_step = 0
        self.temp = self.init_temp * math.exp(-self.exp_anneal_rate * self.cur_step)

    def apply_temp(self, model):
        for m in model.mixed_ops():
            m.set_temperature(self.temp)
