from . import prob  # noqa: F401
import os

# Debug flag - can be set via environment variable ARIA_DEBUG
ARIA_DEBUG = os.environ.get("ARIA_DEBUG", "False").lower() in ("true", "1", "yes")
