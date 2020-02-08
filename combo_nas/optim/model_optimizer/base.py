import random

class ModelOptimizer():
    def __init__(self, space):
        self.space = space

    def get_random_index(self, excludes):
        index = random.randint(0, self.space.categorical_size() - 1)
        while index in excludes:
            index = random.randint(0, self.space.categorical_size() - 1)
        return index

    def get_random_params(self, excludes):
        return self.space.get_categorical_params(self.get_random_index(excludes))

    def get_maximums(self, model, size, excludes):
        raise NotImplementedError
