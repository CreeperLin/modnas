import torch
from ...core.nas_modules import NASModule
from ...utils import accuracy

class REINFORCE():
    def __init__(self, config, net):
        self.net = net
        self.batch_size = config.architect.batch_size
        self.baseline = None
        self.baseline_decay_weight = 0.99

    def reward(self, net_info):
        acc1 = net_info['acc1']
        acc_reward = acc1
        total_reward = acc_reward
        return total_reward
    
    def step(self, trn_X, trn_y, val_X, val_y, lr, w_optim, a_optim):
        a_optim.zero_grad()
        grad_batch = []
        reward_batch = []
        net_info_batch = []
        for i in range(self.batch_size):
            logits = self.net.logits(val_X)
            acc1, acc5 = accuracy(logits, val_y, topk=(1, 5))
            net_info = {'acc1':acc1.item(), }
            net_info_batch.append(net_info)
            # calculate reward according to net_info
            reward = self.reward(net_info)
            # loss term
            obj_term = 0
            for m in self.net.mixed_ops():
                if m.arch_param.grad is not None:
                    m.arch_param.grad.data.zero_()
                path_prob = m.get_state('w_path_f')
                smpl = m.get_state('s_path_f')
                path_prob_f = path_prob.index_select(-1, smpl)
                obj_term = obj_term + torch.log(path_prob_f)
            loss = -obj_term
            # backward
            loss.backward()
            # take out gradient dict
            grad_list = []
            for m in self.net.mixed_ops():
                grad_list.append(m.arch_param.grad.data.clone())
            grad_batch.append(grad_list)
            reward_batch.append(reward)

        # update baseline function
        avg_reward = sum(reward_batch) / self.batch_size
        if self.baseline is None:
            self.baseline = avg_reward
        else:
            self.baseline += self.baseline_decay_weight * (avg_reward - self.baseline)
        # assign gradients
        for idx, m in enumerate(self.net.mixed_ops()):
            m.arch_param.grad.data.zero_()
            for j in range(self.batch_size):
                m.arch_param.grad.data += (reward_batch[j] - self.baseline) * grad_batch[j][idx]
            m.arch_param.grad.data /= self.batch_size
        # apply gradients
        a_optim.step()
        pass