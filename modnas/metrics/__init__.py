from ..registry.metrics import register, get_builder, build, register_as
from .base import MetricsBase
from . import predefined


def build_metrics_all(mt_configs, estim=None, logger=None):
    """Build Metrics from configs."""
    metrics = {}
    if mt_configs is None:
        mt_configs = {}
    MetricsBase.set_estim(estim)
    if not isinstance(mt_configs, dict):
        mt_configs = {'default': mt_configs}
    for mt_name, mt_conf in mt_configs.items():
        mt = build(mt_conf, logger=logger)
        metrics[mt_name] = mt
    MetricsBase.set_estim(None)
    return metrics
