"""Format translators for the ARIA toolkit.

Modules are imported lazily so optional dependencies do not prevent ARIA from
shipping a usable translator package.
"""

from importlib import import_module
from typing import List

__all__: List[str] = []


def _register(name: str, optional: bool = False) -> None:
    try:
        module = import_module(f".{name}", __name__)
    except ImportError:
        if optional:
            return
        raise
    globals()[name] = module
    __all__.append(name)


for required in (
    "dimacs2smt",
    "cnf2smt",
    "cnf2lp",
    "qbf2smt",
    "smt2c",
    "sygus2smt",
    "wcnf2z3",
    "fzn2omt",
):
    _register(required)

for optional in ("smt2sympy",):
    _register(optional, optional=True)
