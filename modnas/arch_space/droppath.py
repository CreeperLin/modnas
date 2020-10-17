"""DropPath."""
from .ops import DropPath_


def update_drop_path_prob(model, drop_path_prob, epoch, tot_epochs):
    """Update DropPath probability."""
    drop_prob = drop_path_prob * epoch / tot_epochs
    for module in model.modules():
        if isinstance(module, DropPath_):
            module.p = drop_prob
