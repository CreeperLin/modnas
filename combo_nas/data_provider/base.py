import logging

class DataProviderBase():
    def __init__(self, logger):
        self.logger = logger or logging.getLogger('data_provider')

    def get_next_train_batch(self,):
        return next(self.get_train_iter())

    def get_next_valid_batch(self,):
        return next(self.get_valid_iter())

    def get_train_iter(self,):
        pass

    def get_valid_iter(self,):
        pass

    def reset_train_iter(self,):
        pass

    def reset_valid_iter(self,):
        pass

    def get_num_train_batch(self,):
        pass

    def get_num_valid_batch(self,):
        pass
