from functools import partial
from combo_nas.arch_space.predefined.mobilenetv2 import MobileInvertedConv, MobileNetV2
from combo_nas.arch_space import register_as
from combo_nas.contrib.arch_space.elastic.spatial import ElasticSpatialGroup, conv2d_rank_weight_l1norm_fan_in
from combo_nas.contrib.arch_space.elastic.sequential import ElasticSequentialGroup
from combo_nas.core.param_space import ArchParamCategorical

class MobileNetV2ElasticSpatialConverter():
    def __init__(self, model, expansion_range=[1, 2, 4, 6], search=False):
        self.model = model
        self.first = False
        self.last_conv = None
        self.last_bn = None
        self.is_search = search
        self.expansion_range = expansion_range

    def __call__(self, slot, *args, fix_first=True, **kwargs):
        if not self.first:
            self.first = True
            if fix_first:
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
        g = ElasticSpatialGroup([last_conv, last_bn, dw_conv, dw_bn], [pw_conv],
                                max_width=slot.kwargs['C'],
                                rank_fn=lambda m=pw_conv: conv2d_rank_weight_l1norm_fan_in(m))
        if self.is_search:
            def on_update_handler(param):
                g.set_width(param.value())
            param_choice = [slot.chn_in * e for e in self.expansion_range]
            p = ArchParamCategorical(param_choice, on_update=on_update_handler)
        self.last_conv = pw_conv
        self.last_bn = pw_bn
        return ent


class MobileNetV2ElasticSequentialConverter():
    def __init__(self, model, repeat_range=[1, 2, 3, 4], search=False):
        self.model = model
        self.first = False
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
                p = ArchParamCategorical(self.repeat_range, on_update=partial(on_update_handler, g))

    def __call__(self, slot, *args, **kwargs):
        if not self.first:
            self.first = True
        ent = MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)
        return ent


class MobileNetV2ElasticSpatial(MobileNetV2):
    def get_predefined_augment_converter(self):
        return MobileNetV2ElasticSpatialConverter(self, search=False)

    def get_predefined_search_converter(self):
        return MobileNetV2ElasticSpatialConverter(self, search=True)


class MobileNetV2ElasticSequential(MobileNetV2):
    def get_predefined_augment_converter(self):
        return MobileNetV2ElasticSequentialConverter(self, search=False)

    def get_predefined_search_converter(self):
        return MobileNetV2ElasticSequentialConverter(self, search=True)


@register_as('ImageNet-MobileNetV2-E-Spatial')
def imagenet_mobilenetv2(chn_in, n_classes, cfgs=None, **kwargs):
    default_cfgs = [
        # t, c, n, s,
        [0, 32, 1, 2],
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]
    if cfgs is None:
        cfgs = default_cfgs
    return MobileNetV2ElasticSpatial(chn_in, n_classes, cfgs, **kwargs)


@register_as('CIFAR-MobileNetV2-E-Spatial')
def cifar_mobilenetv2(chn_in, n_classes, cfgs=None, **kwargs):
    default_cfgs = [
        # t, c, n, s,
        [0, 32, 1, 1], # stride = 1
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 1], # stride = 1
        [6, 320, 1, 1]
    ]
    if cfgs is None:
        cfgs = default_cfgs
    return MobileNetV2ElasticSpatial(chn_in, n_classes, cfgs, **kwargs)


@register_as('MobileNetV2-E-Spatial')
def mobilenetv2_spatial(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2ElasticSpatial(chn_in, n_classes, cfgs, **kwargs)


@register_as('MobileNetV2-E-Sequential')
def mobilenetv2_sequential(chn_in, n_classes, cfgs, **kwargs):
    return MobileNetV2ElasticSequential(chn_in, n_classes, cfgs, **kwargs)
