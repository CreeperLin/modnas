"""Implementation of Metrics interface."""
import logging


class MetricsBase(object):
    """Base Metrics class."""

    cur_estim = None

    def __init__(self, logger):
        super().__init__()
        self.logger = logging.getLogger('metrics') if logger is None else logger
        self.estim = MetricsBase.get_estim()

    def __call__(self, *args, **kwargs):
        """Compute metrics."""
        raise NotImplementedError

    @staticmethod
    def get_estim():
        """Get current Estimator."""
        return MetricsBase.cur_estim

    @staticmethod
    def set_estim(estim):
        """Set current Estimator."""
        MetricsBase.cur_estim = estim
