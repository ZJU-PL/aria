"""
Transition system (Abstraction of programs)
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import z3
from aria.efmc.utils import ctx_simplify


class TransitionSystem(object):
    """
    Transition system supporting multiple data types including QF_FP.

    Supports: integers, reals, bit-vectors, booleans, floating-point, and arrays.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Core components
        self.variables: List[z3.ExprRef] = []  # Current state variables
        self.prime_variables: List[z3.ExprRef] = []  # Next state variables
        self.all_variables: List[z3.ExprRef] = []  # variables + prime_variables
        self.trans: Optional[z3.ExprRef] = None  # Transition relation
        self.init: Optional[z3.ExprRef] = None  # Initial condition
        self.post: Optional[z3.ExprRef] = None  # Post condition
        self.invariants: List[z3.ExprRef] = []  # Additional invariants/lemmas
        self.initialized: bool = False

        # Type flags
        self.has_bv: bool = False
        self.has_int: bool = False
        self.has_real: bool = False
        self.has_array: bool = False
        self.has_bool: bool = False
        self.has_fp: bool = False
        self.signedness: str = "unsigned"

        # Handle keyword arguments if provided
        if kwargs:
            self._init_from_kwargs(**kwargs)

    def _init_from_kwargs(self, **kwargs: Any) -> None:
        """Initialize the transition system from keyword arguments."""
        allowed_keys = {
            "variables",
            "prime_variables",
            "init",
            "trans",
            "post",
            "invariants",
        }
        for key, value in kwargs.items():
            if key not in allowed_keys:
                raise ValueError(f"Unexpected argument: {key}")
            setattr(
                self,
                key,
                (
                    list(value)
                    if key in ["variables", "prime_variables"]
                    else (
                        list(value)
                        if key == "invariants" and isinstance(value, (list, tuple))
                        else value
                    )
                ),
            )

        self.all_variables = self.variables + self.prime_variables
        self._detect_types()
        self.initialized = True

    def _detect_types(self) -> None:
        """Detect variable types from the first variable."""
        if not self.variables:
            return

        sample_var = self.variables[0]
        if z3.is_int(sample_var):
            self.has_int = True
        elif z3.is_real(sample_var):
            self.has_real = True
        elif z3.is_bv(sample_var):
            self.has_bv = True
        elif z3.is_bool(sample_var):
            self.has_bool = True
        elif z3.is_fp(sample_var):
            self.has_fp = True
        elif z3.is_array(sample_var):
            self.has_array = True
        else:
            raise NotImplementedError(f"Unsupported variable type: {sample_var.sort()}")

    def set_signedness(self, ty: str) -> None:
        assert self.has_bv
        if ty == "signed":
            self.signedness = "signed"
        elif ty == "unsigned":
            self.signedness = "unsigned"
        else:
            raise NotImplementedError

    def add_post(self, exp: z3.ExprRef) -> None:
        """Update/initialize set.post (to be verified)"""
        self.post = exp

    def __repr__(self) -> str:
        print(self.all_variables)
        print(self.init)
        print(self.trans)
        print(self.post)
        return " "

    def from_z3_cnts(self, ts: List[z3.ExprRef]) -> None:
        """Initialize from Z3 constraints list [variables, init, trans, post]"""
        self.all_variables, self.init, self.trans, self.post = ts

        # Separate variables and prime variables based on naming convention
        for var in self.all_variables:
            var_name = str(var)
            if var_name.endswith(("!", "'", "_p")):
                self.prime_variables.append(var)
            else:
                self.variables.append(var)

        self._detect_types()
        self.initialized = True

    def analyze_and_simplify(self) -> None:
        """Simplify the problem?"""
        self.trans = ctx_simplify(self.trans)  # ctx_simplify can be slow

    def _get_sort_signature(self) -> str:
        """Get sort signature for invariant function."""
        if self.has_int:
            return "z3.IntSort()"
        elif self.has_real:
            return "z3.RealSort()"
        elif self.has_bv:
            return f"z3.BitVecSort({self.variables[0].sort().size()})"
        elif self.has_bool:
            return "z3.BoolSort()"
        elif self.has_fp:
            fp_sort = self.variables[0].sort()
            return f"z3.FPSort({fp_sort.ebits()}, {fp_sort.sbits()})"
        else:
            raise NotImplementedError(f"Unsupported sort for variables")

    def _get_logic_string(self) -> str:
        """Get SMT-LIB logic string based on variable types."""
        if self.has_array:
            return "ALL"  # Arrays require more complex logic
        elif self.has_fp:
            return "FP"
        elif self.has_bv:
            return "UFBV"
        elif self.has_int:
            return "UFLIA"
        elif self.has_real:
            return "UFLRA"
        elif self.has_fp:
            return "UFFP"
        else:
            return "ALL"

    def to_chc_constraints(self) -> z3.ExprRef:
        """Convert to CHC constraints with invariant function."""
        if self.has_array:
            raise NotImplementedError("Array support not yet implemented")

        s = z3.SolverFor("HORN")
        sort_sig = self._get_sort_signature()

        # Build invariant function signature
        inv_sig = f"z3.Function('inv', {', '.join([sort_sig] * len(self.variables))}, z3.BoolSort())"
        inv = eval(inv_sig)

        # Add CHC constraints
        s.add(z3.ForAll(self.variables, z3.Implies(self.init, inv(self.variables))))
        s.add(
            z3.ForAll(
                self.all_variables,
                z3.Implies(
                    z3.And(inv(self.variables), self.trans), inv(self.prime_variables)
                ),
            )
        )
        s.add(z3.ForAll(self.variables, z3.Implies(inv(self.variables), self.post)))

        return z3.And(s.assertions())

    def to_chc_str(self) -> str:
        """Convert to CHC format"""
        assert self.initialized
        z3_expr = self.to_chc_constraints()
        return f"(set-logic HORN)\n{z3_expr.sexpr()}\n(check-sat)\n"

    def to_smt2(self) -> str:
        """Convert to SMT2 format"""
        assert self.initialized
        z3_expr = self.to_chc_constraints()
        logic = self._get_logic_string()
        return f"(set-logic {logic})\n{z3_expr.sexpr()}\n(check-sat)\n"

    def to_z3_cnts(self) -> List[z3.ExprRef]:
        return self.all_variables, self.init, self.trans, self.post
