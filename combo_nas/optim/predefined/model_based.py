from ..base import CategoricalSpaceOptim
from .. import cost_model, model_optimizer

class ModelBasedOptim(CategoricalSpaceOptim):
    def __init__(self, space, cost_model_config, model_optimizer_config,
                 greedy_e=0.05, n_next_pts=32,):
        super().__init__(space)
        self.cost_model = cost_model.build(cost_model_config['type'], space=space,
                                           **cost_model_config.get('args', {}))
        self.model_optimizer = model_optimizer.build(model_optimizer_config['type'], space=space,
                                                     **model_optimizer_config.get('args', {}))
        self.n_next_pts = n_next_pts
        self.greedy_e = greedy_e
        self.train_x = []
        self.train_y = []
        self.next_xs = []
        self.next_pt = 0
        self.train_ct = 0

    def _next(self):
        while self.next_pt < len(self.next_xs):
            params = self.next_xs[self.next_pt]
            if not self.is_visited_params(params):
                break
            self.next_pt += 1
        if self.next_pt >= len(self.next_xs) - int(self.greedy_e * self.n_next_pts):
            params = self.get_random_params()
        self.set_visited_params(params)
        return params

    def step(self, estim):
        inputs, results = estim.get_last_results()
        for inp, res in zip(inputs, results):
            self.train_x.append(inp)
            self.train_y.append(res)

        if len(self.train_x) < self.n_next_pts * (self.train_ct + 1):
            return

        self.cost_model.fit(self.train_x, self.train_y)
        self.next_xs = self.model_optimizer.get_maximums(
            self.cost_model, self.n_next_pts, self.visited)
        self.next_pt = 0
        self.train_ct += 1
