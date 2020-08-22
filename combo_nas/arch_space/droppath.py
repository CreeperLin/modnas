import torch.nn as nn
from .slot import Slot
from .ops import DropPath_, Identity

def update_drop_path_prob(model, drop_path_prob, epoch, tot_epochs):
    drop_prob = drop_path_prob * epoch / tot_epochs
    for module in model.modules():
        if isinstance(module, DropPath_):
            module.p = drop_prob


def apply_drop_path(model):
    def apply(slot):
        ent = slot.ent
        if slot.fixed or ent is None or isinstance(ent, Identity):
            return
        ent = nn.Sequential(
            ent,
            DropPath_()
        )
        slot.set_entity(ent)
    Slot.apply_all(apply, gen=Slot.gen_slots_model(model))
