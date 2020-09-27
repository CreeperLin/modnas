"""Create and access new Registry as modules."""
import sys
from ..utils.registration import get_registry_utils


class RegistryModule():
    """Registry as a module."""

    def __init__(self, fullname):
        registry_name = fullname.split('.')[-1]
        self.__package__ = fullname
        self.__path__ = fullname.split('.')
        self.__name__ = registry_name
        self.reg_path, self.register, self.get_builder, self.build, self.register_as = get_registry_utils(registry_name)


class RegistryImporter():
    """Create new Registry using import hooks (PEP 302)."""

    def find_module(self, fullname, path=None):
        """Handle registry imports."""
        if fullname.startswith('combo_nas.registry'):
            return self

    def load_module(self, fullname):
        """Create and find registry by import path."""
        mod = sys.modules.get(fullname, RegistryModule(fullname))
        sys.modules[fullname] = mod
        return mod


sys.meta_path.append(RegistryImporter())
