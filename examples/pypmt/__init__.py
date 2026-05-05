
import sys
from typing import Tuple, Union

VERSION: Tuple[Union[int, str], ...] = (0, 2, 0)
__version__ = ".".join(str(x) for x in VERSION)

sys.modules.setdefault("pypmt", sys.modules[__name__])
