from .default_estimator import DefaultEstimator
from .supernet_estimator import SuperNetEstimator
from .subnet_estimator import SubNetEstimator
from .hptune_estimator import HPTuneEstimator

from .. import register_estimator
register_estimator(DefaultEstimator, 'Default')
register_estimator(SuperNetEstimator, 'SuperNet')
register_estimator(SubNetEstimator, 'SubNet')
register_estimator(HPTuneEstimator, 'HPTune')
