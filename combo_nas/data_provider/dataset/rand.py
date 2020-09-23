import torch
from torch.utils.data import TensorDataset
from . import register


def get_data_shape(shape):
    if shape in [None, 'nil', 'None']:
        return []
    elif isinstance(shape, int):
        return [shape]
    return shape


def get_dtype(dtype):
    if dtype == 'float':
        return torch.float32
    elif dtype == 'int':
        return torch.int64
    else:
        return None


def get_random_data(shape, dtype, drange):
    data = torch.Tensor(*shape)
    if drange in [None, 'nil', 'None']:
        data.normal_()
    else:
        data.uniform_(drange[0], drange[1])
    return data.to(dtype=get_dtype(dtype))


@register
def RandData(data_spec, data_size=128):
    data = []
    for dshape, dtype, drange in data_spec:
        data.append(get_random_data([data_size] + get_data_shape(dshape), dtype, drange))
    dset = TensorDataset(*data)
    return dset
