import torch
from torch.utils.data import TensorDataset
from .dataset import register_as

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


@register_as('rand')
def get_rand_dataset(validation, data_spec,
                      trn_size=128, val_size=128):
    trn_dset = val_dset = None
    trn_data = []
    for dshape, dtype, drange in data_spec:
        trn_data.append(get_random_data([trn_size] + get_data_shape(dshape), dtype, drange))
    trn_dset = TensorDataset(*trn_data)
    if validation:
        val_data = []
        for dshape, dtype, drange in data_spec:
            val_data.append(get_random_data([val_size] + get_data_shape(dshape), dtype, drange))
        val_dset = TensorDataset(*val_data)
    return trn_dset, val_dset
