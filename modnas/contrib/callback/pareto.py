import matplotlib
from modnas.registry.callback import register
from modnas.registry.callback import OptimumReporter
matplotlib.use('Agg')
from matplotlib import pyplot as plt


@register
class ParetoReporter(OptimumReporter):

    def __init__(self, *args, plot_keys=None, plot_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_keys = plot_keys
        self.plot_args = plot_args

    def report_results(self, ret, estim, optim):
        ret = super().report_results(ret, estim, optim)
        if not self.results or not self.opt_results:
            return ret
        plt.figure()
        plt.title('Pareto optimum')
        plot_keys = self.plot_keys or list(self.results[0][1].keys())[:2]
        if len(plot_keys) < 2:
            self.logger.error('Not enough metrics for pareto plot')
            return ret
        domed_res = [r for r in self.results if r not in self.opt_results]
        vals = [[m.get(k, 0) for _, m in domed_res] for k in plot_keys]
        opt_vals = [[m.get(k, 0) for _, m in self.opt_results] for k in plot_keys]
        plt.scatter(*vals, **(self.plot_args or {}))
        plt.scatter(*opt_vals, **(self.plot_args or {}))
        plot_path = estim.expman.join('plot', 'pareto.png')
        plt.savefig(plot_path)
        self.logger.info('Pareto plot saved to {}'.format(plot_path))
        return ret
