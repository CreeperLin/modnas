import matplotlib
from modnas.registry.callback import register
from modnas.registry.callback import OptimumReporter
matplotlib.use('Agg')
from matplotlib import pyplot as plt


@register
class ParetoReporter(OptimumReporter):

    def __init__(self, *args, plot_keys=None, plot_args=None, plot_intv=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_keys = plot_keys
        self.plot_args = plot_args
        self.plot_intv = plot_intv

    def plot_pareto(self, estim, epoch=None):
        if not self.results or not self.opt_results:
            return
        plt.figure()
        plt.title('Pareto optimum')
        plot_keys = self.plot_keys or list(self.results[0][1].keys())[:2]
        if len(plot_keys) < 2:
            self.logger.error('Not enough metrics for pareto plot')
            return
        domed_res = [r for r in self.results if r not in self.opt_results]
        vals = [[m.get(k, 0) for _, m in domed_res] for k in plot_keys]
        opt_vals = [[m.get(k, 0) for _, m in self.opt_results] for k in plot_keys]
        plt.scatter(*vals, **(self.plot_args or {}))
        plt.scatter(*opt_vals, **(self.plot_args or {}))
        plt.xlabel(plot_keys[0])
        plt.ylabel(plot_keys[1])
        plot_path = estim.expman.join('plot', 'pareto{}.png'.format('' if epoch is None else ('_' + str(epoch))))
        plt.savefig(plot_path)
        self.logger.info('Pareto plot saved to {}'.format(plot_path))

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        if not self.plot_intv is None and (epoch + 1) % self.plot_intv == 0:
            self.plot_pareto(estim, epoch + 1)
        return super().report_epoch(ret, estim, optim, epoch, tot_epochs)

    def report_results(self, ret, estim, optim):
        self.plot_pareto(estim)
        return super().report_results(ret, estim, optim)
