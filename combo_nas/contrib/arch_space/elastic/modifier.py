def get_ori_param(module, name):
    return module._params_ori[name]


def get_ori_buffer(module, name):
    return module._buffers_ori[name]


def get_ori_attr(module, name):
    return module._attrs_ori[name]


def backup_param(module, name):
    if not hasattr(module, '_params_ori'):
        module._params_ori = dict()
    if name in module._params_ori:
        return
    val = module._parameters[name]
    module._params_ori[name] = val


def backup_buffer(module, name):
    if not hasattr(module, '_buffers_ori'):
        module._buffers_ori = dict()
    if name in module._buffers_ori:
        return
    val = module._buffers[name]
    module._buffers_ori[name] = val


def backup_attr(module, name):
    if not hasattr(module, '_attrs_ori'):
        module._attrs_ori = dict()
    if name in module._attrs_ori:
        return
    val = getattr(module, name)
    module._attrs_ori[name] = val


def update_param(module, name, val):
    if not hasattr(module, '_params_ori'):
        return
    if name not in module._params_ori:
        return
    module._params_ori[name] = val


def update_buffer(module, name, val):
    if not hasattr(module, '_buffers_ori'):
        return
    if name not in module._buffers_ori:
        return
    module._buffers_ori[name] = val


def update_attr(module, name, val):
    if not hasattr(module, '_attrs_ori'):
        return
    if name not in module._attrs_ori:
        return
    module._attrs_ori[name] = val


def restore_param(module, name):
    if not hasattr(module, '_params_ori'):
        return
    if name not in module._params_ori:
        return
    val = module._params_ori.pop(name)
    module._parameters[name] = val


def restore_buffer(module, name):
    if not hasattr(module, '_buffers_ori'):
        return
    if name not in module._buffers_ori:
        return
    val = module._buffers_ori.pop(name)
    module._buffers[name] = val


def restore_attr(module, name):
    if not hasattr(module, '_attrs_ori'):
        return
    if name not in module._attrs_ori:
        return
    val = module._attrs_ori.pop(name)
    object.__setattr__(module, name, val)


def modify_param(module, name, value):
    backup_param(module, name)
    module._parameters[name] = value


def modify_buffer(module, name, value):
    backup_buffer(module, name)
    module._buffers[name] = value


def modify_attr(module, name, value):
    backup_attr(module, name)
    object.__setattr__(module, name, value)


def restore_module_parameters(module):
    if hasattr(module, '_params_ori'):
        module._parameters.update(module._params_ori)
        module._params_ori.clear()


def restore_module_buffers(module):
    if hasattr(module, '_buffers_ori'):
        module._buffers.update(module._buffers_ori)
        module._buffers_ori.clear()


def restore_module_attrs(module):
    if hasattr(module, '_attrs_ori'):
        module.__dict__.update(module._attrs_ori)
        module._attrs_ori.clear()


def restore_module_states(module):
    restore_module_parameters(module)
    restore_module_buffers(module)
    restore_module_attrs(module)
