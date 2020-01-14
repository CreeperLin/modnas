
class MetricsBase(object):
    def __init__(self):
        super().__init__()
    
    def compute(self, *args, **kwargs):
        raise NotImplementedError
