from functools import partial
from combo_nas.arch_space.predefined.mobilenetv2 import MobileNetV2PredefinedConstructor
from combo_nas.arch_space.construct import register as register_constructor
from combo_nas.arch_space.export import register as register_exporter
from combo_nas.contrib.arch_space.elastic.spatial import ElasticSpatialGroup,\
    conv2d_rank_weight_l1norm_fan_in, conv2d_rank_weight_l1norm_fan_out, batchnorm2d_rank_weight_l1norm
from combo_nas.contrib.arch_space.elastic.sequential import ElasticSequentialGroup
from combo_nas.core.param_space import ArchParamSpace, ArchParamCategorical


@register_exporter
class MobileNetV2ElasticArchDescExporter():
    def __init__(self, fix_first=True, max_stage_depth=4):
        self.fix_first = fix_first
        self.max_stage_depth = max_stage_depth

    def __call__(self, model):
        arch_desc = []
        max_stage_depth = self.max_stage_depth
        if self.fix_first:
            arch_desc.append(None)
        params = {k: p.value() for k, p in ArchParamSpace.named_params()}
        seq_values = [v for k, v in params.items() if k.startswith('seq')]
        n_sequential = len(seq_values)
        spa_values = [v for k, v in params.items() if k.startswith('spa')]
        if not len(spa_values):
            spa_values = [6] * sum([len(btn) for btn in model.bottlenecks if len(btn) > 1])
        for i, spa in enumerate(spa_values):
            cur_seq_idx = i // max_stage_depth
            seq = seq_values[cur_seq_idx] if cur_seq_idx < len(seq_values) else cur_seq_idx
            exp = spa if cur_seq_idx >= n_sequential or i % max_stage_depth < seq else -1
            desc = 'NIL' if exp == -1 else 'MB3E{}'.format(exp)
            arch_desc.append(desc)
        return arch_desc


@register_constructor
class MobileNetV2ElasticSpatialConverter(MobileNetV2PredefinedConstructor):
    def __init__(self, fix_first=True, expansion_range=[1, 3, 6], rank_fn='l1_fan_in', search=True):
        super().__init__()
        self.fix_first = fix_first
        self.first = False
        self.last_conv = None
        self.last_bn = None
        self.is_search = search
        self.expansion_range = expansion_range
        self.rank_fn = rank_fn
        self.spa_group_cnt = 0

    def convert(self, slot):
        if not self.first:
            self.first = True
            if self.fix_first:
                ent = super().convert(slot)
                self.last_conv = ent[3]
                self.last_bn = ent[4]
                return ent
            else:
                self.last_conv = self.model.conv_first[0]
                self.last_bn = self.model.conv_first[1]
        ent = super().convert(slot)
        expansion_range = self.expansion_range
        if isinstance(expansion_range[0], list):
            expansion_range = expansion_range[self.spa_group_cnt]
        self.add_spa_group(slot, ent, expansion_range)
        return ent

    def add_spa_group(self, slot, ent, expansion_range):
        max_width = slot.kwargs['C']
        if any([e * slot.chn_in > max_width for e in expansion_range]):
            raise ValueError('invalid expansion_range: {} max: {}'.format(expansion_range, max_width))
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
            rank_fn = partial(conv2d_rank_weight_l1norm_fan_in, pw_conv)
        elif self.rank_fn == 'l1_fan_out':
            rank_fn = partial(conv2d_rank_weight_l1norm_fan_out, pw_conv)
        elif self.rank_fn == 'bn_l1':
            rank_fn = partial(batchnorm2d_rank_weight_l1norm, pw_bn)
        else:
            raise ValueError('unsupported rank function')
        g = ElasticSpatialGroup([last_conv, last_bn, dw_conv, dw_bn], [pw_conv], max_width=max_width, rank_fn=rank_fn)
        if self.is_search:

            def on_update_handler(chn_in, param):
                g.set_width(chn_in * param.value())

            param_choice = [e for e in expansion_range]
            _ = ArchParamCategorical(param_choice, name='spa', on_update=partial(on_update_handler, slot.chn_in))
        self.last_conv = pw_conv
        self.last_bn = pw_bn
        self.spa_group_cnt += 1


@register_constructor
class MobileNetV2ElasticSequentialConverter(MobileNetV2PredefinedConstructor):
    def __init__(self, repeat_range=[1, 2, 3, 4], search=True):
        super().__init__()
        self.is_search = search
        self.repeat_range = repeat_range

    def __call__(self, model):
        model = super().__call__(model)
        self.make_sequential_groups(model)
        return model

    def make_sequential_groups(self, model):
        bottlenecks = model.bottlenecks
        for i, btn in enumerate(bottlenecks):
            if len(list(btn)) <= 1:
                continue
            repeat_range = self.repeat_range
            if isinstance(repeat_range[0], list):
                repeat_range = repeat_range[i]
            self.add_seq_group(btn, repeat_range)

    def add_seq_group(self, bottleneck, repeat_range):
        blocks = list(bottleneck)
        max_stage_depth = len(blocks)
        g = ElasticSequentialGroup(*blocks)
        if any([r > max_stage_depth for r in repeat_range]):
            raise ValueError('invalid repeat_range: {} max: {}'.format(repeat_range, max_stage_depth))
        if self.is_search:

            def on_update_handler(group, param):
                group.set_depth(param.value())

            _ = ArchParamCategorical(repeat_range, name='seq', on_update=partial(on_update_handler, g))


@register_constructor
class MobileNetV2ElasticConverter(MobileNetV2ElasticSpatialConverter, MobileNetV2ElasticSequentialConverter):
    __call__ = MobileNetV2ElasticSequentialConverter.__call__
    convert = MobileNetV2ElasticSpatialConverter.convert

    def __init__(self, search=True, spatial_kwargs=None, sequential_kwargs=None):
        MobileNetV2ElasticSpatialConverter.__init__(self, search=search, **(spatial_kwargs or {}))
        MobileNetV2ElasticSequentialConverter.__init__(self, search=search, **(sequential_kwargs or {}))
