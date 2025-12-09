from enum import Enum, auto


class VarState(Enum):
    TRUE_VAL = auto()
    FALSE_VAL = auto()
    UNASSIGNED = auto()
    IRRELEVANT = auto()


class ClauseState(Enum):
    ACTIVE = auto()
    PASSIVE = auto()
