from functools import partial
from combo_nas.arch_space import register, ops
from combo_nas.arch_space.predefined.darts import DARTSLikeNet, build_from_config

class StacNASNet(DARTSLikeNet):

    def get_genotype_search_converter(self, primitives_map=None):
        primitives_map = {
            'MAX': ['AVG', 'MAX', 'NIL'],
            'AVG': ['AVG', 'MAX', 'NIL'],
            'SC3': ['SC3', 'SC5', 'NIL'],
            'SC5': ['SC3', 'SC5', 'NIL'],
            'DC3': ['DC3', 'DC5', 'NIL'],
            'DC5': ['DC3', 'DC5', 'NIL'],
        } if primitives_map is None else primitives_map
        predefined_convert_fn = self.get_predefined_search_converter()
        def convert_fn(slot, gene, *args, **kwargs):
            if isinstance(gene, list): gene = gene[0]
            if gene in ['IDT', 'NIL']:
                ent = ops.build(gene, slot.chn_in, slot.chn_out, slot.stride)
            else:
                kwargs['primitives'] = primitives_map[gene]
                ent = predefined_convert_fn(slot, *args, **kwargs)
            return ent
        return convert_fn

register(partial(build_from_config, darts_cls=StacNASNet), 'StacNAS')
