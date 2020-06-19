from .default_estimator import DefaultEstimator
from .supernet_estimator import SuperNetEstimator
from .subnet_estimator import SubNetEstimator
from .hptune_estimator import HPTuneEstimator
from .unified_estimator import UnifiedEstimator

from .. import register
register(DefaultEstimator, 'Default')
register(SuperNetEstimator, 'SuperNet')
register(SubNetEstimator, 'SubNet')
register(HPTuneEstimator, 'HPTune')
register(UnifiedEstimator, 'Unified')
