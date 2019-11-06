from .tuner import Tuner

class XGBoostTuner(Tuner):
    """XGBoost Tuner"""
    def __init__(self, space):
        super(XGBoostTuner, self).__init__(space)
        self.counter = 0

    def next(self):
        if self.counter >= len(self.space): return None
        index = self.counter
        self.counter = self.counter + 1
        return self.space.get(index)

    def has_next(self):
        return self.counter < len(self.space)
    
    def update(self, inputs, result):
        pass

    def load_history(self, data_set):
        pass

    def __getstate__(self):
        return {"counter": self.counter}

    def __setstate__(self, state):
        self.counter = state['counter']
