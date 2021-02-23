from functools import partial
import importlib
from ...registry.backend import register


register(partial(importlib.import_module, 'modnas.backend.predefined.torch'), 'torch')
register(partial(importlib.import_module, 'modnas.backend.predefined.tensorflow'), 'tensorflow')
