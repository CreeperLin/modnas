from functools import partial
from combo_nas.arch_space.predefined.mobilenetv2 import MobileInvertedConv, MobileNetV2
from combo_nas.arch_space import register_as
from combo_nas.contrib.arch_space.elastic.spatial import ElasticSpatialGroup,\
    conv2d_rank_weight_l1norm_fan_in, conv2d_rank_weight_l1norm_fan_out, batchnorm2d_rank_weight_l1norm
from combo_nas.contrib.arch_space.elastic.sequential import ElasticSequentialGroup
from combo_nas.core.param_space import ArchParamCategorical

class MobileNetV2ElasticSpatialConverter():
    def __init__(self, model, fix_first=True, expansion_range=[1, 3, 6], rank_fn='l1_fan_in', search=False):
        self.model = model
        self.fix_first = fix_first
        self.first = False
        self.last_conv = None
        self.last_bn = None
        self.is_search = search
        self.expansion_range = expansion_range
        self.rank_fn = rank_fn

    def __call__(self, slot, *args, **kwargs):
        if not self.first:
            self.first = True
            if self.fix_first:
                ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
                self.last_conv = ent[3]
                self.last_bn = ent[4]
                return ent
            else:
                self.last_conv = self.model.conv_first[0]
                self.last_bn = self.model.conv_first[1]
        ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
        num_blocks = len(ent)
        if num_blocks != 8:
            # not tested
            last_conv = self.last_conv
            last_bn = self.last_bn
            dw_conv = ent[0]
            dw_bn = ent[1]
            pw_conv = ent[3]
            pw_bn = ent[4]
        else:
            last_conv = ent[0]
            last_bn = ent[1]
            dw_conv = ent[3]
            dw_bn = ent[4]
            pw_conv = ent[6]
            pw_bn = ent[7]
        if self.rank_fn is None or self.rank_fn == 'none':
            rank_fn = None
        elif self.rank_fn == 'l1_fan_in':
            rank_fn = lambda m=pw_conv: conv2d_rank_weight_l1norm_fan_in(m)
        elif self.rank_fn == 'l1_fan_out':
            rank_fn = lambda m=pw_conv: conv2d_rank_weight_l1norm_fan_out(m)
        elif self.rank_fn == 'bn_l1':
            rank_fn = lambda m=pw_bn: batchnorm2d_rank_weight_l1norm(m)
        else:
            raise ValueError('unsupported rank function')
        g = ElasticSpatialGroup([last_conv, last_bn, dw_conv, dw_bn], [pw_conv],
                                max_width=slot.kwargs['C'],
                                rank_fn=rank_fn)
        if self.is_search:
            def on_update_handler(chn_in, param):
                g.set_width(chn_in * param.value())
            param_choice = [e for e in self.expansion_range]
            p = ArchParamCategorical(param_choice, name='spa', on_update=partial(on_update_handler, slot.chn_in))
        self.last_conv = pw_conv
        self.last_bn = pw_bn
        return ent


class MobileNetV2ElasticSequentialConverter():
    def __init__(self, model, repeat_range=[1, 2, 3, 4], search=False):
        self.model = model
        self.is_search = search
        self.repeat_range = repeat_range
        self.make_sequential_groups()

    def make_sequential_groups(self):
        bottlenecks = self.model.bottlenecks
        for btn in bottlenecks:
            blocks = list(btn)
            if len(blocks) <= 1:
                continue
            g = ElasticSequentialGroup(*blocks)
            if self.is_search:
                def on_update_handler(group, param):
                    group.set_depth(param.value())
                p = ArchParamCategorical(self.repeat_range, name='seq', on_update=partial(on_update_handler, g))

    def __call__(self, slot, *args, **kwargs):
        ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
        return ent


class MobileNetV2ElasticConverter(MobileNetV2ElasticSpatialConverter, MobileNetV2ElasticSequentialConverter):
    def __init__(self, model, search=False, spatial_kwargs=None, sequential_kwargs=None):
        MobileNetV2ElasticSpatialConverter.__init__(self, model, search=search, **(spatial_kwargs or {}))
        MobileNetV2ElasticSequentialConverter.__init__(self, model, search=search, **(sequential_kwargs or {}))

    def __call__(self, slot, *args, **kwargs):
        return MobileNetV2ElasticSpatialConverter.__call__(self, slot, *args, **kwargs)


class MobileNetV2ElasticSpatial(MobileNetV2):
    def get_predefined_augment_converter(self, *args, **kwargs):
        return MobileNetV2ElasticSpatialConverter(self, search=False, *args, **kwargs)

    def get_predefined_search_converter(self, *args, **kwargs):
        return MobileNetV2ElasticSpatialConverter(self, search=True, *args, **kwargs)


class MobileNetV2ElasticSequential(MobileNetV2):
    def get_predefined_augment_converter(self, *args, **kwargs):
        return MobileNetV2ElasticSequentialConverter(self, search=False, *args, **kwargs)

    def get_predefined_search_converter(self, *args, **kwargs):
        return MobileNetV2ElasticSequentialConverter(self, search=True, *args, **kwargs)


class MobileNetV2Elastic(MobileNetV2):
    def get_predefined_augment_converter(self, *args, **kwargs):
        return MobileNetV2ElasticConverter(self, search=False, *args, **kwargs)

    def get_predefined_search_converter(self, *args, **kwargs):
        return MobileNetV2ElasticConverter(self, search=True, *args, **kwargs)


@register_as('MobileNetV2-E-Spatial')
def mobilenetv2_spatial(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2ElasticSpatial(chn_in, n_classes, cfgs, **kwargs)


@register_as('MobileNetV2-E-Sequential')
def mobilenetv2_sequential(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2ElasticSequential(chn_in, n_classes, cfgs, **kwargs)


@register_as('MobileNetV2-E')
def mobilenetv2_elastic(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2Elastic(chn_in, n_classes, cfgs, **kwargs)
