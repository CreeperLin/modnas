""" arch_optim base """

class ArchOptimBase():
    def __init__(self, config, net):
        self.net = net

    def state_dict(self):
        return {}
    
    def load_state_dict(self, sd):
        pass
    
    def has_next(self):
        pass
    
    def next(self, batch_size):
        pass

    def step(self, estim):
        pass

    def update(self, estim):
        pass