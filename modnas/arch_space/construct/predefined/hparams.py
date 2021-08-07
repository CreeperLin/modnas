"""Hyperparameter constructor."""
from typing import Dict, List, Union
from modnas.registry.construct import register
from modnas.core.params import Numeric, Categorical


@register
class DefaultHParamSpaceConstructor():
    """Constructor that generates parameters from config."""

    def __init__(self, params: Union[Dict, List]) -> None:
        if isinstance(params, dict):
            self.params = params.items()
        elif isinstance(params, list):
            self.params = [(None, p) for p in params]

    def __call__(self, model: None) -> None:
        """Run constructor."""
        del model
        for k, v in self.params:
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                _ = Numeric(low=v[0][0], high=v[0][1], name=k)
            elif isinstance(v, list):
                _ = Categorical(choices=v, name=k)
            else:
                raise ValueError('support categorical and numeric hparams only')
