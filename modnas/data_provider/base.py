"""Base DataProvider."""
from modnas.utils.logging import get_logger


class DataProviderBase():
    """Base DataProvider class."""

    logger = get_logger('data_provider')

    def get_next_train_batch(self):
        """Return the next train batch."""
        return next(self.get_train_iter())

    def get_next_valid_batch(self):
        """Return the next validate batch."""
        return next(self.get_valid_iter())

    def get_train_iter(self):
        """Return train iterator."""
        raise NotImplementedError

    def get_valid_iter(self):
        """Return validate iterator."""
        raise NotImplementedError

    def reset_train_iter(self):
        """Reset train iterator."""
        raise NotImplementedError

    def reset_valid_iter(self):
        """Reset validate iterator."""
        raise NotImplementedError

    def get_num_train_batch(self):
        """Return number of train batches in current epoch."""
        raise NotImplementedError

    def get_num_valid_batch(self):
        """Return number of validate batches in current epoch."""
        raise NotImplementedError
