import pickle
import itertools
import matplotlib
from modnas.registry.estim import register
from modnas.registry.estim import SubNetEstim
matplotlib.use('Agg')
from matplotlib import pyplot as plt


@register
class SubNetStatsEstim(SubNetEstim):
    def __init__(self, axis_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subnet_results = []
        self.axis_list = axis_list

    def step(self, params):
        ret = super().step(params)
        self.subnet_results.append((params, ret))
        return ret

    def run(self, optim):
        ret = super().run(optim)
        subnet_results = self.subnet_results
        axis_list = self.axis_list
        if axis_list is None:
            metrics = list(subnet_results[0][1].keys())
            axis_list = list(itertools.combinations(metrics, r=2))
        self.logger.info('subnet stats: {} axis: {}'.format(len(subnet_results), axis_list))
        for i, axis in enumerate(axis_list):
            plt.figure(i)
            axis_str = '-'.join(axis)
            plt.title('subnet metrics: {}'.format(axis_str))
            values = [[res[1][ax] for res in subnet_results] for ax in axis]
            plt.scatter(values[0], values[1])
            plt.xlabel(axis[0])
            plt.ylabel(axis[1])
            plt.savefig(self.expman.join('plot', 'subnet_{}.png'.format(axis_str)))
        result_path = self.expman.join('output', 'subnet_results.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(subnet_results, f)
            self.logger.info('subnet results saved to {}'.format(result_path))
        return ret
