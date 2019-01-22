def merge_dicts(a, b, path=None):
    """
    Merges b into a.

    Taken from https://stackoverflow.com/a/7205107/1116842
    """
    if path is None:
        path = []
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a
