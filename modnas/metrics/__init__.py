from ..registry.metrics import register, get_builder, build, register_as
from .base import MetricsBase


def build_metrics_all(mt_configs, estim=None, logger=None):
    """Build Metrics from configs."""
    metrics = {}
    if mt_configs is None:
        mt_configs = {}
    MetricsBase.set_estim(estim)
    if not isinstance(mt_configs, dict):
        mt_configs = {'default': mt_configs}
    for mt_name, mt_conf in mt_configs.items():
        if isinstance(mt_conf, str):
            mt_conf = {'type': mt_conf}
        mt_type = mt_conf['type']
        mt_args = mt_conf.get('args', {})
        mt = build(mt_type, logger, **mt_args)
        metrics[mt_name] = mt
    MetricsBase.set_estim(None)
    return metrics
