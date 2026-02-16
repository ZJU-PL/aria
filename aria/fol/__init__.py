"""
First-Order Logic (FOL) reasoning and theorem proving.

This package provides tools for working with first-order logic:

- `miniprover/`: An educational automated theorem prover for first-order logic.
  Implements a complete proof system based on sequent calculus that is guaranteed
  to find proofs for provable formulas (though may not terminate for unprovable ones).

For more details on the theorem prover, see `miniprover/README.md`.

Example:
    >>> from aria.fol.miniprover import prover, language
    >>> # Use the theorem prover
"""

from . import miniprover

__all__ = ["miniprover"]
