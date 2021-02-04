def singleton(cls):
    inst = []

    def get_instance(*args, **kwargs):
        if not inst:
            inst.append(cls(*args, **kwargs))
        return inst[0]
    return get_instance
