"""Utility functions for Spyro synthesis."""


def find_linenum_starts_with(lines, target, start=0):
    """Find the line number where a line starts with target string."""
    for i, line in enumerate(lines[start:]):
        if 0 == line.find(target):
            return start+i

    return -1


def find_linenum_with(lines, target, start=0):
    """Find the line number where a line contains target string."""
    for i, line in enumerate(lines[start:]):
        if line.find(target) >= 0:
            return start+i

    return -1


def sum_dict(d1, d2):
    """Sum values from two dictionaries, combining matching keys."""
    d = d1.copy()
    for k, v in d2.items():
        if k in d.keys():
            d[k] += v
        else:
            d[k] = v

    return d


def max_dict(d1, d2):
    """Take maximum values from two dictionaries, combining matching keys."""
    d = d1.copy()
    for k, v in d2.items():
        if k in d.keys():
            d[k] = max(d[k], v)
        else:
            d[k] = v

    return d


def union_dict(d1, d2):
    """Union two dictionaries, keeping values from d1 for duplicate keys."""
    d = d1.copy()
    for k, v in d2.items():
        if k not in d.keys():
            d[k] = v

    return d
