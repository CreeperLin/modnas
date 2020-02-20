
class MetricsBase(object):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def compute(self, *args, **kwargs):
        raise NotImplementedError
