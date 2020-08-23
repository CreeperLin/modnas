from .base import DataProviderBase
from . import register


class DefaultDataProvider(DataProviderBase):
    def __init__(self, train_loader, valid_loader, logger=None):
        super().__init__(logger)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_iter = None
        self.valid_iter = None
        self.no_valid_warn = True
        self.reset_train_iter()
        self.reset_valid_iter()

    def get_next_train_batch(self):
        if self.train_loader is None:
            self.logger.error('no train loader')
            return None
        try:
            trn_batch = next(self.get_train_iter())
        except StopIteration:
            self.reset_train_iter()
            trn_batch = next(self.get_train_iter())
        return trn_batch

    def get_next_valid_batch(self):
        if self.valid_loader is None:
            if self.no_valid_warn:
                self.logger.warning('no valid loader, returning training batch instead')
                self.no_valid_warn = False
            return self.get_next_train_batch()
        try:
            val_batch = next(self.get_valid_iter())
        except StopIteration:
            self.reset_valid_iter()
            val_batch = next(self.get_valid_iter())
        return val_batch

    def get_train_iter(self):
        return self.train_iter

    def get_valid_iter(self):
        return self.valid_iter

    def reset_train_iter(self):
        self.train_iter = None if self.train_loader is None else iter(self.train_loader)

    def reset_valid_iter(self):
        self.valid_iter = None if self.valid_loader is None else iter(self.valid_loader)

    def get_num_train_batch(self, epoch):
        return 0 if self.train_loader is None else len(self.train_loader)

    def get_num_valid_batch(self, epoch):
        return 0 if self.valid_loader is None else len(self.valid_loader)


register(DefaultDataProvider, 'Default')
