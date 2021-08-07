"""Score model optimum finder."""
import random
from collections import OrderedDict
from typing import Set


class ModelOptim():
    """Score model optimum finder class."""

    def __init__(self, space):
        self.space = space

    def get_random_index(self, excludes: Set[int]) -> int:
        """Return random categorical index from search space."""
        index = random.randint(0, self.space.categorical_size() - 1)
        while index in excludes:
            index = random.randint(0, self.space.categorical_size() - 1)
        return index

    def get_random_params(self, excludes: Set[int]) -> OrderedDict:
        """Return random categorical parameters from search space."""
        return self.space.get_categorical_params(self.get_random_index(excludes))

    def get_optimums(self, model, size, excludes):
        """Return optimums in score model."""
        raise NotImplementedError
