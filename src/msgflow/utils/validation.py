import inspect


def is_subclass_of(obj, cls):
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, cls)


def is_builtin_type(obj):
    builtin_types = (
        str, int, float, bool, list, dict, tuple, set,
        type(None)
    )
    return isinstance(obj, builtin_types)