from .default import DefaultEstimator
from .supernet import SuperNetEstimator
from .subnet import SubNetEstimator
from .hptune import HPTuneEstimator
from .unified import UnifiedEstimator

from .. import register
register(DefaultEstimator, 'Default')
register(SuperNetEstimator, 'SuperNet')
register(SubNetEstimator, 'SubNet')
register(HPTuneEstimator, 'HPTune')
register(UnifiedEstimator, 'Unified')
