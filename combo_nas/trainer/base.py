from ..utils import DummyWriter
import logging


class TrainerBase():
    def __init__(self, logger=None, writer=None):
        if logger is None:
            logger = logging.getLogger('Trainer')
        self.logger = logger
        if writer is None:
            writer = DummyWriter()
        self.writer = writer

    def train_epoch(self):
        pass

    def valid_epoch(self):
        pass

    def train_step(self):
        pass

    def valid_step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass
