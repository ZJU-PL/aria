import sys
from types import ModuleType

if "six" not in sys.modules:
    six = ModuleType("six")
    sys.modules["six"] = six

if "six.moves" not in sys.modules:
    moves = ModuleType("six.moves")
    sys.modules["six.moves"] = moves

    import configparser
    moves.configparser = configparser
    moves.xrange = range