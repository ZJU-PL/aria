"""Internal branch and return-value logic for package instrumentation tests."""


def classify_score(value: int) -> int:
    if value > 10:
        return value * 2
    return value - 1


class Formatter:
    @staticmethod
    def decorate(prefix: str, score: int) -> str:
        if score > 15:
            return prefix + ":high"
        return prefix + ":low"


def expanded_score(left: int, right: int, scale: int = 1) -> int:
    return (left + right) * scale


class PropertyBox:
    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def doubled(self) -> int:
        return self.value * 2

    @doubled.setter
    def doubled(self, new_value: int) -> None:
        self.value = new_value // 2
