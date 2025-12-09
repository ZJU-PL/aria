from functools import partial
from typing import Any, Callable, Dict

from multipledispatch import dispatch

namespace: Dict[str, Any] = dict()

dispatch = partial(dispatch, namespace=namespace)
