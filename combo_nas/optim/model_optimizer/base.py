import random

class ModelOptimizer():
    def __init__(self, space):
        self.space = space

    def get_random_param(self, excludes):
        index = random.randint(0, self.space.categorical_size())
        while index in excludes:
            index = random.randint(0, self.space.categorical_size())
        return self.space.get_categorical_params(index)

    def get_maximums(self, model, size, excludes):
        raise NotImplementedError
