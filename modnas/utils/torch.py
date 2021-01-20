import torch


def param_count(model, factor=0, divisor=1000):
    """Return number of model parameters."""
    return sum(p.data.nelement() for p in model.parameters()) / divisor**factor


def param_size(model, factor=0, divisor=1024):
    """Return size of model parameters."""
    return 4 * param_count(model) / divisor**factor


def clear_bn_running_statistics(model):
    """Clear BatchNorm running statistics."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()


def recompute_bn_running_statistics(model, trainer, num_batch=100, clear=True):
    """Recompute BatchNorm running statistics."""
    if clear:
        clear_bn_running_statistics(model)
    is_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(num_batch):
            try:
                trn_X, _ = trainer.get_next_train_batch()
            except StopIteration:
                break
            model(trn_X)
            del trn_X
    if not is_training:
        model.eval()
