"""Dispatch module for multiple dispatch functionality."""

from functools import partial
from typing import Any

from multipledispatch import dispatch

namespace: dict[str, Any] = {}

dispatch = partial(dispatch, namespace=namespace)
