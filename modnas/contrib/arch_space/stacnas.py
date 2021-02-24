from modnas.registry.arch_space import build
from modnas.registry.construct import register
from modnas.registry.construct import DefaultMixedOpConstructor, DefaultRecursiveArchDescConstructor


@register
class StacNASArchDescSearchConstructor(DefaultRecursiveArchDescConstructor, DefaultMixedOpConstructor):
    def __init__(self, *args, arch_desc, primitives_map=None, **kwargs):
        DefaultRecursiveArchDescConstructor.__init__(self, arch_desc=arch_desc)
        DefaultMixedOpConstructor.__init__(self, *args, **kwargs)
        primitives_map = {
            'MAX': ['AVG', 'MAX', 'NIL'],
            'AVG': ['AVG', 'MAX', 'NIL'],
            'SC3': ['SC3', 'SC5', 'NIL'],
            'SC5': ['SC3', 'SC5', 'NIL'],
            'DC3': ['DC3', 'DC5', 'NIL'],
            'DC5': ['DC3', 'DC5', 'NIL'],
        } if primitives_map is None else primitives_map
        self.primitives_map = primitives_map

    def convert(self, slot, desc):
        desc = desc[0] if isinstance(desc, list) else desc
        if desc in ['IDT', 'NIL']:
            ent = build(desc, slot)
        else:
            self.primitives = self.primitives_map[desc]
            ent = DefaultMixedOpConstructor.convert(self, slot)
        return ent
