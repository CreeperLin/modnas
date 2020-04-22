import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import torch.nn.functional as F
from combo_nas.estimator import register_as
from combo_nas.estimator.predefined.supernet_estimator import SuperNetEstimator
from combo_nas.core.param_space import ArchParamSpace

@register_as('ParamStatsSuperNet')
class ParamStatsEstimator(SuperNetEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs = []

    def record_probs(self):
        self.probs.append([F.softmax(a.detach(), dim=-1).cpu().numpy() for a in self.model.arch_param_tensor()])

    def search_epoch(self, epoch, optim):
        self.record_probs()
        return super().search_epoch(epoch, optim)

    def search(self, optim):
        ret = super().search(optim)
        self.record_probs()
        probs = self.probs
        n_alphas = len(probs[0])
        n_epochs = len(probs)
        self.logger.info('arch param stats: epochs: {} alphas: {}'.format(n_epochs, n_alphas))
        epochs = list(range(n_epochs))
        save_probs = []
        for i, alpha in enumerate(ArchParamSpace.tensor_params()):
            plt.figure(i)
            plt.title('alpha: {}'.format(i))
            prob = np.array([p[i] for p in probs])
            alpha_dim = prob.shape[1]
            for a in range(alpha_dim):
                plt.plot(epochs, prob[:, a])
            legends = list(alpha.modules())[0].primitive_names()
            plt.legend(legends)
            plt.savefig(self.expman.join('plot', 'prob_{}.png'.format(i)))
            save_probs.append(prob)
        probs_path = self.expman.join('output', 'probs.pkl')
        with open(probs_path, 'wb') as f:
            pickle.dump(save_probs, f)
            self.logger.info('arch param tensor probs saved to {}'.format(probs_path))
        return ret
