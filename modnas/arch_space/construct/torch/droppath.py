"""DropPath constructor."""
import torch
from modnas.arch_space.ops import DropPath, Identity
from modnas.core.event import event_on
from .default import DefaultSlotTraversalConstructor
from modnas.registry.construct import register
from modnas.utils import copy_members
from modnas.arch_space.slot import Slot
from torch.nn.modules.container import Sequential
from torch.nn.modules.module import Module
from typing import Optional


@register
class DropPathConstructor(DefaultSlotTraversalConstructor):
    """Constructor that applies DropPath on Slot modules."""

    def __init__(self, *args, drop_prob=0.1, skip_exist=False, **kwargs) -> None:
        super().__init__(*args, skip_exist=skip_exist, **kwargs)
        self.drop_prob = drop_prob
        self.transf = DropPathTransformer()

    def __call__(self, model: Module) -> Module:
        """Run constructor."""
        super().__call__(model)

        def drop_prob_update(*args, epoch=None, tot_epochs=None, **kwargs):
            self.transf.set_prob(self.drop_prob * epoch / tot_epochs)
            self.transf(model)

        event_on('before:TrainerBase.train_epoch', drop_prob_update)
        return model

    def convert(self, slot: Slot) -> Optional[Sequential]:
        """Return module with DropPath."""
        ent = slot.get_entity()
        if ent is None or isinstance(ent, Identity):
            return
        new_ent = torch.nn.Sequential(ent, DropPath())
        copy_members(new_ent, ent, excepts=['forward', 'modules', 'named_modules'])
        return new_ent


@register
class DropPathTransformer(DefaultSlotTraversalConstructor):
    """Transformer that update DropPath probability."""

    def __init__(self, *args, skip_exist=False, **kwargs) -> None:
        super().__init__(*args, skip_exist=skip_exist, **kwargs)
        self.prob = None

    def set_prob(self, prob: float) -> None:
        """Set DropPath probability."""
        self.prob = prob

    def convert(self, slot: Slot) -> None:
        """Apply DropPath probability on Slot module."""
        ent = slot.get_entity()
        if ent is None or isinstance(ent, Identity):
            return
        for m in ent.modules():
            if isinstance(m, DropPath):
                m.drop_prob = self.prob
