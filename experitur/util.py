def callable_to_name(obj):
    if callable(obj):
        return "{}.{}".format(obj.__module__, obj.__name__)

    if isinstance(obj, list):
        return [callable_to_name(x) for x in obj]

    if isinstance(obj, dict):
        return {callable_to_name(k): callable_to_name(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(callable_to_name(x) for x in obj)

    return obj
