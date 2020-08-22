import logging

class MetricsBase(object):
    cur_estim = None

    def __init__(self, logger):
        super().__init__()
        self.logger = logging.getLogger('metrics') if logger is None else logger
        self.estim = MetricsBase.get_estim()

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_estim():
        return MetricsBase.cur_estim

    @staticmethod
    def set_estim(estim):
        MetricsBase.cur_estim = estim
