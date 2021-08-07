from modnas.registry.metrics import build
from modnas.registry import SPEC_TYPE
from .base import MetricsBase
from typing import Dict, Optional, Any


def build_metrics_all(mt_configs: Optional[SPEC_TYPE], estim: Optional[Any] = None) -> Dict[str, MetricsBase]:
    """Build Metrics from configs."""
    metrics = {}
    MetricsBase.set_estim(estim)
    if mt_configs is None:
        mt_configs = {}
    if not isinstance(mt_configs, dict):
        mt_configs = {'default': mt_configs}
    for mt_name, mt_conf in mt_configs.items():
        mt = build(mt_conf)
        metrics[mt_name] = mt
    MetricsBase.set_estim(None)
    return metrics
