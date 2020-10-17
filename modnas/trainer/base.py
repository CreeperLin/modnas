"""Base Trainer."""
from ..utils import DummyWriter
import logging


class TrainerBase():
    """Base Trainer class."""

    def __init__(self, logger=None, writer=None):
        if logger is None:
            logger = logging.getLogger('Trainer')
        self.logger = logger
        if writer is None:
            writer = DummyWriter()
        self.writer = writer

    def train_epoch(self):
        """Train for one epoch."""
        raise NotImplementedError

    def valid_epoch(self):
        """Validate for one epoch."""
        raise NotImplementedError

    def train_step(self):
        """Train for one step."""
        raise NotImplementedError

    def valid_step(self):
        """Validate for one step."""
        raise NotImplementedError

    def state_dict(self):
        """Return current states."""
        return {}

    def load_state_dict(self, sd):
        """Resume states."""
        raise NotImplementedError
