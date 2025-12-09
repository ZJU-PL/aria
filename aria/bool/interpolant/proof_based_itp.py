"""
Ken McMillan's method for computing interpolants from resolution proofs

"Interpolation and SAT-based Model Checking" by Kenneth L. McMillan, CAV 2003.

This module implements McMillan's algorithm for computing Craig interpolants
from resolution proofs of unsatisfiability. The method takes a resolution proof
that shows A ∧ B is unsatisfiable and computes an interpolant I such that:
1. A implies I
2. I ∧ B is unsatisfiable
3. I only contains variables common to A and B

The problem is, how to extract the resolution proof, e.g., from Z3, pysat, pySMT, or CVC5?. Or, should  we use the a resolution-based proof system by ourself (which can be slow)?
"""

import z3
from typing import Dict, List, Set, Tuple, Union, Optional, Callable
import contextlib
