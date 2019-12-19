""" arch_optim base """

class ArchOptimBase():
    def __init__(self, config):
        self.config = config

    def state_dict(self):
        return {}
    
    def load_state_dict(self, sd):
        pass
    
    def has_next(self):
        pass
    
    def _next(self):
        pass
    
    def next(self, batch_size):
        batch = []
        for i in range(batch_size):
            if self.has_next():
                batch.append(self._next())
        return batch

    def step(self, estim):
        pass

    def update(self, estim):
        pass