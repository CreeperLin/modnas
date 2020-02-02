""" optim base """

class OptimBase():
    def __init__(self, space):
        self.space = space

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
        for _ in range(batch_size):
            if not self.has_next():
                break
            batch.append(self._next())
        return batch

    def step(self, estim):
        pass

    def update(self, estim):
        pass
