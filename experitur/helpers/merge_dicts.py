def merge_dicts(a, b):
    """
    Recursively merge b into a, overwriting existing values in a.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a
