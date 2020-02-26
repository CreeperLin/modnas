import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from combo_nas.arch_space.ops import register_as

@register_as('MaskConv')
class MaskedConv(nn.Conv2d):
    def __init__(self, chn_in, chn_out, kernel_size, stride, padding, dilation=1, groups=1, bias=False, mask=None):
        nn.Module.__init__(self)
        self.chn_in = chn_in
        self.chn_out = chn_out
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.Tensor(chn_out)) if bias else None
        self.kernel_shape = (self.chn_out, self.chn_in, self.kernel_size[0], self.kernel_size[1])
        self.weight = nn.Parameter(torch.Tensor(*self.kernel_shape))
        if mask is None:
            mask = torch.ones(*self.kernel_shape)
        elif len(mask.shape) <= 2:
            mask = mask.repeat(self.chn_out, self.chn_in, 1, 1)
        if mask.shape != self.kernel_shape:
            raise ValueError('invalid mask shape: {}'.format(mask.shape))
        self.mask = mask
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        kernel = self.weight * self.mask
        return F.conv2d(x, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)


@register_as('IdxConv')
class IndexedConv(nn.Conv2d):
    def __init__(self, chn_in, chn_out, kernel_size, stride, padding, dilation=1, groups=1, bias=False, indices=None):
        nn.Module.__init__(self)
        self.chn_in = chn_in
        self.chn_out = chn_out
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.Tensor(chn_out)) if bias else None
        self.kernel_shape = (self.chn_out, self.chn_in, self.kernel_size[0], self.kernel_size[1])
        if indices is None:
            self.indices = None
            self.weight = nn.Parameter(torch.Tensor(*self.kernel_shape))
            self.reset_parameters()
            return
        if not isinstance(indices, torch.LongTensor):
            if isinstance(indices, (tuple, list)):
                indices = torch.LongTensor(indices)
            else:
                raise ValueError('invalid indices type: {}'.format(type(indices)))
        if len(indices.shape) != 2:
            raise ValueError('invalid indices shape: {}'.format(indices.shape))
        self.indices = indices
        n_weights = self.indices.shape[1]
        if n_weights <= 0 or n_weights > kernel_size[0] * kernel_size[1]:
            raise ValueError('invalid number of indices: {}'.format(n_weights))
        self.weight = nn.Parameter(torch.Tensor(self.chn_out, self.chn_in, n_weights))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.indices is None:
            kernel = self.weight
        else:
            kernel = torch.zeros(self.kernel_shape).to(device=self.weight.device)
            kernel[:, :, self.indices[0], self.indices[1]] = torch.flatten(self.weight, start_dim=2)
        return F.conv2d(x, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)


@register_as('IC3[x]')
def cross_conv_3x3(*args, **kwargs):
    indices = [
        [0,0,1,2,2],
        [0,2,1,0,2],
    ]
    return IndexedConv(*args, **kwargs, indices=indices)


@register_as('IC3[+]')
def plus_conv_3x3(*args, **kwargs):
    indices = [
        [0,1,1,1,2],
        [1,0,1,2,1],
    ]
    return IndexedConv(*args, **kwargs, indices=indices)
