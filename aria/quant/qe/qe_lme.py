"""Quantifier Elimination via Lazy Model Enumeration (LME-QE)"""

from collections.abc import Iterable
from typing import Any, List, Optional, cast

import z3

from ._projection import get_projection_vars, normalize_vars
from aria.utils.z3.expr import get_atoms, negate


def eval_predicates(m: z3.ModelRef, preds: Iterable[z3.BoolRef]) -> List[z3.BoolRef]:
    """Evaluate predicates in a model and return their truth values."""
    res: List[z3.BoolRef] = []
    for p in preds:
        if z3.is_true(m.eval(p)):
            res.append(p)
        elif z3.is_false(m.eval(p)):
            res.append(cast(z3.BoolRef, negate(p)))
    return res


def process_model(
    phi: Any,
    qvars: List[z3.ExprRef],
    preds: Iterable[z3.BoolRef],
    shared_models: Iterable[z3.BoolRef],
) -> Optional[z3.BoolRef]:
    """Process a single model for quantifier elimination."""
    s = z3.Solver()
    s.add(phi)
    for model in shared_models:
        s.add(negate(model))
    if s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        proj = z3.Tactic("qe2")(z3.Exists(qvars, minterm)).as_expr()
        return cast(z3.BoolRef, proj)
    return None

def qelim_exists_lme(
    phi: Any, qvars: Any, *, keep_vars: Optional[Any] = None
) -> z3.BoolRef:
    """Eliminate existential quantifiers using lazy model enumeration.

    When ``keep_vars`` is provided, every free variable in ``phi`` that is not
    explicitly kept is projected away together with ``qvars``.
    """
    normalized_qvars = normalize_vars(qvars)
    normalized_keep_vars = normalize_vars(keep_vars) if keep_vars is not None else None
    projection_vars = get_projection_vars(phi, normalized_qvars, normalized_keep_vars)

    s = z3.Solver()
    s.add(phi)
    res: List[z3.BoolRef] = []
    preds = cast(Iterable[z3.BoolRef], get_atoms(cast(z3.BoolRef, phi)))
    qe_for_conjunction = z3.Tactic("qe2")
    while s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        proj = cast(
            z3.BoolRef,
            qe_for_conjunction(z3.Exists(projection_vars, minterm)).as_expr(),
        )
        res.append(proj)
        s.add(negate(proj))
    return cast(z3.BoolRef, z3.simplify(z3.Or(res)))


def test_qe():
    """Test quantifier elimination with a sample formula."""
    # x, y, z = z3.BitVecs("x y z", 16)
    x, y, z = z3.Reals("x y z")
    fml = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))  # x: 4, y: 1
    qf = qelim_exists_lme(fml, [x, y])
    print(qf)


if __name__ == "__main__":
    test_qe()
