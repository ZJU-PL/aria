# Python 3.13 compatibility fix for six.moves
# Python 3.13 removed internal APIs that six.moves relies on
# This patch fixes six.moves imports before PySMT tries to use them
import sys

if sys.version_info >= (3, 13):
    try:
        import six
        import configparser
        import types

        # Create a real module for six.moves and register it in sys.modules
        # This allows "import six.moves.*" to work
        if 'six.moves' not in sys.modules:
            moves_module = types.ModuleType('six.moves')
            sys.modules['six.moves'] = moves_module
            six.moves = moves_module

        # Add commonly needed modules from six.moves
        if not hasattr(six.moves, 'configparser'):
            six.moves.configparser = configparser
            sys.modules['six.moves.configparser'] = configparser

        # xrange was removed in Python 3, it's just range now
        if not hasattr(six.moves, 'xrange'):
            six.moves.xrange = range

        # cStringIO was removed in Python 3, use io.StringIO instead
        if not hasattr(six.moves, 'cStringIO'):
            import io
            six.moves.cStringIO = io.StringIO
    except ImportError:
        pass  # six not installed yet, will fail later with a clearer error

from . import prob  # noqa: F401
import os

# Debug flag - can be set via environment variable ARIA_DEBUG
ARIA_DEBUG = os.environ.get("ARIA_DEBUG", "False").lower() in ("true", "1", "yes")
