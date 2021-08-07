"""Experiment file manager."""
import os
import time
from .logging import get_logger
from typing import Optional


class ExpManager():
    """Experiment file manager class."""

    logger = get_logger('exp_manager')

    def __init__(self, name: str, root_dir: str = 'exp', subdir_timefmt: Optional[str] = None) -> None:
        if subdir_timefmt is None:
            root_dir = os.path.join(root_dir, name)
        else:
            root_dir = os.path.join(root_dir, name, time.strftime(subdir_timefmt, time.localtime()))
        self.root_dir = os.path.realpath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.logger.info('exp dir: {}'.format(self.root_dir))

    def subdir(self, *args) -> str:
        """Return subdir in current root dir."""
        subdir = os.path.join(self.root_dir, *args)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def join(self, *args) -> str:
        """Join root dir and subdir path."""
        return os.path.join(self.subdir(*args[:-1]), args[-1])
