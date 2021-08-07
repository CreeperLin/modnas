"""Default registry."""
from modnas.utils.logging import get_logger
from typing import Any


class Registry():
    """Registry class."""

    logger = get_logger('registry')

    def __init__(self, allow_replace: bool = False) -> None:
        self.allow_replace = allow_replace
        self._reg_class = {}

    def get_full_path(self, reg_path: str, reg_id: str) -> str:
        """Return full registration path."""
        return '{}.{}'.format(reg_path, reg_id)

    def get_reg_name(self, reg_path: str, reg_id: str) -> str:
        """Return proper registration name."""
        name = self.get_full_path(reg_path, reg_id)
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass: Any, reg_path: str, reg_id: str) -> None:
        """Register a component class."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id in self._reg_class:
            self.logger.warning('re-register id: {}'.format(reg_id))
            if not self.allow_replace:
                raise ValueError('Cannot re-register id: {}'.format(reg_id))
        self._reg_class[reg_id] = regclass
        self.logger.debug('registered: {}'.format(reg_id))

    def get(self, reg_path: str, reg_id: str) -> Any:
        """Return registered class by name."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id not in self._reg_class:
            raise ValueError('id \'{}\' not found in registry'.format(reg_id))
        return self._reg_class[reg_id]


registry = Registry()
