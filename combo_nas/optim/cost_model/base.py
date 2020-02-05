import numpy as np

class CostModel():
    def __init__(self, space):
        self.space = space

    def fit(self, inputs, results):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def _process_input(self, inp):
        ret = []
        for n, v in inp.items():
            p = self.space.get_param(n)
            one_hot = [0] * len(p)
            one_hot[p.get_index(v)] = 1
            ret.extend(one_hot)
        return ret

    def to_feature(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [self._process_input(inp) for inp in inputs]
        return np.array(inputs)

    def to_target(self, results):
        return np.array(results)
