"""Implementation of Metrics interface."""
from modnas.utils.logging import get_logger


class MetricsBase():
    """Base Metrics class."""

    logger = get_logger('metrics')
    cur_estim = None

    def __init__(self):
        self.estim = MetricsBase.get_estim()

    def __call__(self, *args, **kwargs):
        """Return metrics output."""
        raise NotImplementedError

    @staticmethod
    def get_estim():
        """Get current Estimator."""
        return MetricsBase.cur_estim

    @staticmethod
    def set_estim(estim):
        """Set current Estimator."""
        MetricsBase.cur_estim = estim
