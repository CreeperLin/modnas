"""Pipeline Estimator."""
import traceback
import queue
import multiprocessing as mp
from ..base import EstimBase
from modnas.registry.estim import register
from modnas.utils.wrapper import run


def _mp_step_runner(conn, step_conf):
    ret = run(**step_conf)
    conn.send(ret)


def _mp_runner(step_conf):
    ctx = mp.get_context('spawn')
    p_con, c_con = ctx.Pipe()
    proc = ctx.Process(target=_mp_step_runner, args=(c_con, step_conf.to_dict()))
    proc.start()
    proc.join()
    if not p_con.poll(0):
        return None
    return p_con.recv()


def _default_runner(step_conf):
    return run(**step_conf)


@register
class PipelineEstim(EstimBase):
    """Pipeline Estimator class."""

    def __init__(self, *args, use_multiprocessing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = _mp_runner if use_multiprocessing else _default_runner

    def step(self, step_conf):
        """Return results from single pipeline process."""
        try:
            return self.runner(step_conf)
        except RuntimeError:
            self.logger.info('pipeline step failed with error: {}'.format(traceback.format_exc()))
        return None

    def run(self, optim):
        """Run Estimator routine."""
        del optim
        logger = self.logger
        config = self.config
        pipeconf = config.pipeline
        pending = queue.Queue()
        for pn in pipeconf.keys():
            pending.put(pn)
        finished = set()
        ret_values, ret = dict(), None
        while not pending.empty():
            pname = pending.get()
            pconf = pipeconf.get(pname)
            dep_sat = True
            for dep in pconf.get('depends', []):
                if dep not in finished:
                    dep_sat = False
                    break
            if not dep_sat:
                pending.put(pname)
                continue
            pconf['name'] = pconf.get('name', pname)
            for inp_kw, inp_idx in pconf.get('inputs', {}).items():
                keys = inp_idx.split('.')
                inp_val = ret_values
                for k in keys:
                    if not inp_val or k not in inp_val:
                        raise RuntimeError('input key {} not found in return {}'.format(inp_idx, ret_values))
                    inp_val = inp_val[k]
                pconf[inp_kw] = inp_val
            logger.info('pipeline: running {}'.format(pname))
            ret = self.step(pconf)
            ret_values[pname] = ret
            logger.info('pipeline: finished {}, results={}'.format(pname, ret))
            finished.add(pname)
        logger.info('pipeline: all finished')
        return ret_values
