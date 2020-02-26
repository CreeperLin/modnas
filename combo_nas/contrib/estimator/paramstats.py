import os
import numpy as np
from matplotlib import pyplot as plt 

import torch.nn.functional as F
import combo_nas.estimator
from combo_nas.estimator.predefined.supernet_estimator import SuperNetEstimator

@combo_nas.estimator.register_as('ParamStatsSuperNet')
class ParamStatsEstimator(SuperNetEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs = []

    def search_epoch(self, epoch, optim):
        self.probs.append([F.softmax(a.detach(), dim=-1).cpu().numpy() for a in self.model.arch_param_tensor()])
        super().search_epoch(epoch, optim)

    def search(self, optim):
        ret = super().search(optim)
        probs = self.probs
        mixed_ops = list(self.model.mixed_ops())
        n_alphas = len(probs[0])
        n_epochs = len(probs)
        self.logger.info('arch param stats: epochs: {} alphas: {}'.format(n_epochs, n_alphas))
        epochs = list(range(n_epochs))
        for aidx in range(n_alphas):
            plt.figure(aidx)
            plt.title('alpha: {}'.format(aidx))
            prob = np.array([p[aidx] for p in probs])
            alpha_dim = prob.shape[1]
            for a in range(alpha_dim):
                plt.plot(epochs, prob[:, a])
            legends = mixed_ops[aidx].ops
            plt.legend(legends)
            plt.savefig(os.path.join(self.expman.plot_path(), 'prob_{}.png'.format(aidx)))
        return ret
