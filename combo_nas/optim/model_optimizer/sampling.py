from .base import ModelOptimizer


class RandomSamplingModelOptimizer(ModelOptimizer):
    def __init__(self, space, n_iter=1000):
        super().__init__(space)
        self.n_iter = n_iter

    def get_maximums(self, model, size, excludes):
        smpl_pts = [self.get_random_index(excludes) for _ in range(self.n_iter)]
        smpl_val = model.predict([self.space.get_categorical_params(i) for i in smpl_pts])
        topk_idx = smpl_val.argsort()[::-1][:size]
        return [smpl_pts[i] for i in topk_idx]
