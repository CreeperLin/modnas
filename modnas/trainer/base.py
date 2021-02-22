"""Base Trainer."""
import logging
from ..utils import DummyWriter
from ..core.event import event_hooked_subclass


@event_hooked_subclass
class TrainerBase():
    """Base Trainer class."""

    def __init__(self, logger=None, writer=None):
        if logger is None:
            logger = logging.getLogger('Trainer')
        self.logger = logger
        if writer is None:
            writer = DummyWriter()
        self.writer = writer

    def init(self, model, config=None):
        raise NotImplementedError

    def model_input(self, data):
        """Return model input."""
        return data[:-1], {}

    def model_output(self, *args, data=None, model=None, attr=None, **kwargs):
        """Return model output."""
        model_fn = model if attr is None else getattr(model, attr)
        if data is not None:
            args, kwargs = self.model_input(data)
        return model_fn(*args, **kwargs)

    def loss(self, output=None, data=None, model=None):
        """Return loss."""
        return None

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
