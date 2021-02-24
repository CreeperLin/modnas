from ..base import EstimBase
from modnas.registry.estim import register, build
from ...registry.dist_remote import build as build_remote
from ...registry.dist_worker import build as build_worker


@register
class DistributedEstim(EstimBase):
    def __init__(self, estim_conf, remote_conf, worker_conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_main = self.config.get('main', False)
        estim_comp_keys = [
            'expman',
            'constructor',
            'exporter',
            'model',
            'writer',
            'logger',
        ]
        estim_comp = {k: getattr(self, k) for k in estim_comp_keys}
        self.estim = build(estim_conf, config=estim_conf, **estim_comp)
        self.estim_step = self.estim.step
        self.estim.step = self.step
        if self.is_main:
            self.remote = build_remote(remote_conf)
        else:
            self.worker = build_worker(worker_conf)

    def step(self, params):
        def on_done(ret):
            self.logger.debug('Dist main: params: {} ret: {}'.format(params, ret))
            self.estim.step_done(params, ret, self.estim.get_arch_desc())

        def on_failed(ret):
            self.estim.step_done(params, 0, 0)

        if self.is_main:
            self.remote.call('step', params, on_done=on_done, on_failed=on_failed)
            return
        ret = self.estim_step(params)
        self.logger.debug('Dist worker: params: {} ret: {}'.format(params, ret))
        return ret

    def run(self, optim):
        if self.is_main:
            return self.estim.run(optim)
        return self.worker.run(self.estim)
