from functools import wraps, partial


def make_decorator(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0:
            return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapped


def singleton(cls):
    inst = []

    def get_instance(*args, **kwargs):
        if not inst:
            inst.append(cls(*args, **kwargs))
        return inst[0]
    return get_instance
