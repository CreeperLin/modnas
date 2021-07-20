"""Constructed modules."""
from modnas.registry.construct import build as build_constructor
from modnas.registry.arch_space import build as build_module
from modnas.registry.arch_space import register
from modnas.registry import streamline_spec


@register
def Constructed(slot=None, construct=None, module=None):
    """Return a module from constructors."""
    m = None if module is None else build_module(module, slot=slot)
    for con in streamline_spec(construct):
        m = build_constructor(con)(m)
    return m
