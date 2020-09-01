import torch
from ..ops import DropPath_, Identity
from .default import DefaultSlotTraversalConstructor
from . import register


@register
class DropPathConstructor(DefaultSlotTraversalConstructor):
    def convert(self, slot):
        ent = slot.ent
        if ent is None or isinstance(ent, Identity):
            return ent
        return torch.nn.Sequential(ent, DropPath_())


@register
class DropPathTransformer(DefaultSlotTraversalConstructor):
    def __init__(self, prob, total_steps):
        super().__init__()
        self.prob = prob
        self.total_steps = total_steps
        self.step = -1

    def __trigger__(self, model):
        self.step += 1
        self.cur_prob = self.prob * self.step / self.total_steps
        self.__call__(model)

    def convert(self, slot):
        ent = slot.ent
        for m in ent.modules():
            if isinstance(m, DropPath_):
                m.p = self.cur_prob
        return ent
