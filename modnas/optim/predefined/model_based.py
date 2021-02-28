from ..base import CategoricalSpaceOptim
from modnas.registry.cost_model import build as build_cost_model
from modnas.registry.model_optimizer import build as build_model_optimizer
from modnas.registry.optim import register


@register
class ModelBasedOptim(CategoricalSpaceOptim):
    def __init__(self, cost_model_config, model_optimizer_config, greedy_e=0.05, n_next_pts=32, space=None):
        super().__init__(space)
        self.cost_model = build_cost_model(cost_model_config, space=self.space)
        self.model_optimizer = build_model_optimizer(model_optimizer_config,
                                                     space=self.space)
        self.n_next_pts = n_next_pts
        self.greedy_e = greedy_e
        self.train_x = []
        self.train_y = []
        self.next_xs = []
        self.next_pt = 0
        self.train_ct = 0

    def _next(self):
        while self.next_pt < len(self.next_xs):
            index = self.next_xs[self.next_pt]
            if not self.is_visited(index):
                break
            self.next_pt += 1
        if self.next_pt >= len(self.next_xs) - int(self.greedy_e * self.n_next_pts):
            index = self.get_random_index()
        self.set_visited(index)
        return self.space.get_categorical_params(index)

    def step(self, estim):
        inputs, results = estim.get_last_results()
        for inp, res in zip(inputs, results):
            self.train_x.append(inp)
            self.train_y.append(res)
        if len(self.train_x) < self.n_next_pts * (self.train_ct + 1):
            return
        self.cost_model.fit(self.train_x, self.train_y)
        self.next_xs = self.model_optimizer.get_maximums(self.cost_model, self.n_next_pts, self.visited)
        self.next_pt = 0
        self.train_ct += 1
