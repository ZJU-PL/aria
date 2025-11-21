"""Parse an OMT instance"""

import z3
from z3.z3consts import *
from typing import List, Optional, Any


class OMTParser:
    """Currently, we focus on two modes
    1. Single-objective optimization
    2. Multi-objective optimization under the boxed mode (each obj is independent)"""

    def __init__(self) -> None:
        """
        For multi-objective optimization,
        """
        self.assertions: Optional[List[z3.ExprRef]] = None
        # the (possibly transformed) objectives after normalisation (see to_max_obj/to_min_obj)
        self.objectives: List[z3.ExprRef] = []

        # keep legacy flags – callers (e.g. omt_solver.py) currently expect all goals
        # to be converted to "maximise" form.  Setting `to_max_obj=True` preserves
        # that behaviour while still allowing mixed-direction inputs.
        self.to_max_obj: bool = True   # convert every objective to maximise form
        self.to_min_obj: bool = False  # convert every objective to minimise form

        # original_directions[i] is either "max" or "min" indicating the *source*
        # sense of objectives[i] before any normalisation.
        self.original_directions: List[str] = []

        # for convenience in single-objective instances
        self.objective: Optional[z3.ExprRef] = None

        # default off; use logging instead of prints for production use
        self.debug: bool = False

    def parse_with_pysmt(self) -> None:
        """Parse OMT instance using PySMT (not implemented)."""
        # pysmt does not support
        raise NotImplementedError

    def parse_with_z3(self, fml: str, is_file: bool = False) -> None:
        """Parse OMT instance using Z3.

        Args:
            fml: Formula string or file path
            is_file: Whether fml is a file path

        FIXME: Should we convert all the objectives/goals as all "minimize goals" (as Z3 does)?
            (or should we convert them to "maximize goals"?)
            However, the queries can be of the form "max x; min x; max y; min y; ...."
        """
        s = z3.Optimize()
        if is_file:
            s.from_file(fml)
        else:
            s.from_string(fml)
        self.assertions = s.assertions()
        # sanity check for mutually-exclusive normalisation options
        if self.to_min_obj and self.to_max_obj:
            raise ValueError("Cannot set both 'to_min_obj' and 'to_max_obj' to True")

        def _is_bv(expr: z3.ExprRef) -> bool:
            return expr.sort_kind() == z3.Z3_BV_SORT

        def _bvneg(e: z3.ExprRef) -> z3.ExprRef:
            # wrapper to handle BV negation uniformly across bit-widths
            return z3.BVSub(z3.BitVecVal(0, e.size()), e)

        # First collect original expressions and their optimisation sense
        raw_objectives = []
        directions = []  # parallel list of "max" / "min"
        for obj in s.objectives():
            if obj.decl().kind() in (Z3_OP_UMINUS, Z3_OP_BNEG):
                # Z3 converted a (maximize t) into (minimize (- t)) → original was MAX
                directions.append("max")
                raw_objectives.append(obj.children()[0])
            else:
                directions.append("min")
                raw_objectives.append(obj)

        if not raw_objectives:
            raise ValueError("No objectives found in the supplied formula/file")

        # Now perform optional normalisation so that all objectives share the same
        # direction (max or min) expected by downstream code.
        self.objectives = []
        if self.to_max_obj:
            for expr, dirn in zip(raw_objectives, directions):
                if dirn == "max":
                    self.objectives.append(expr)
                else:  # original was MIN – negate appropriately
                    self.objectives.append(_bvneg(expr) if _is_bv(expr) else -expr)
        elif self.to_min_obj:
            for expr, dirn in zip(raw_objectives, directions):
                if dirn == "min":
                    self.objectives.append(expr)
                else:  # original was MAX – negate appropriately
                    self.objectives.append(_bvneg(expr) if _is_bv(expr) else -expr)
        else:
            # keep original direction (no sign flipping)
            self.objectives = list(raw_objectives)

        self.original_directions = directions

        # single-objective convenience handle
        if len(self.objectives) == 1:
            self.objective = self.objectives[0]

        if self.debug:
            import logging
            logger = logging.getLogger(__name__)
            for expr, dirn in zip(self.objectives, self.original_directions):
                logger.debug("objective (%s): %s", dirn, expr)




if __name__ == "__main__":
    a, b, c, d = z3.Ints('a b c d')
    fml = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))
