import os
import sys
import builtins

# Make Z3 library findable at runtime
try:
    bundle_dir = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    if hasattr(builtins, "Z3_LIB_DIRS"):
        builtins.Z3_LIB_DIRS.append(bundle_dir)
    else:
        builtins.Z3_LIB_DIRS = [bundle_dir]
except Exception:
    pass
