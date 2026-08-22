"""Public entry point spanning functions and a class method."""

from .core import Formatter, PropertyBox, classify_score, expanded_score


def entry(value: int, prefix: str) -> str:
    score = classify_score(value)
    rendered = Formatter.decorate(prefix, score)
    if rendered.startswith("a:high"):
        return "hit"
    return "miss"


def expansion_entry(args: list[int], kwargs: dict[str, int]) -> bool:
    score = expanded_score(*args, **kwargs)
    if score > 20:
        return True
    return False


def property_entry(box: PropertyBox, new_value: int) -> bool:
    box.doubled = new_value
    if box.doubled > 10:
        return True
    return False


def lazy_entry(value: int) -> bool:
    from .lazy import lazy_helper

    if lazy_helper(value) > 12:
        return True
    return False
