import importlib
from ..registry.backend import build
from . import predefined

_backend = None

_backend_keys = []


def use(backend, *args, imported=False, **kwargs):
    """Switch to backend by name."""
    global _backend, _backend_keys
    if backend == _backend:
        return
    if imported:
        bk_mod = importlib.import_module(backend)
    else:
        bk_mod = build(backend, *args, **kwargs)
    bk_vars = vars(bk_mod)
    bk_keys = bk_vars.keys()
    ns = globals()
    for k in _backend_keys:
        ns.pop(k, None)
    for k in bk_keys:
        ns[k] = bk_vars[k]
    _backend_keys = bk_keys
    _backend = backend


def backend():
    """Return name of current backend."""
    return _backend
