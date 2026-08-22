"""Lazily imported helper used to test runtime import instrumentation."""


def lazy_helper(value: int) -> int:
    if value > 5:
        return value * 3
    return value
