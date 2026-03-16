"""
Enhanced TransitionSystem that supports:
1. Multiple invariants
2. Mixed variable types (int, real, bool, bv, array)
3. Flexible number of variables and prime variables
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict
import z3


class NewTransitionSystem:
    """Enhanced transition system supporting multiple invariants and mixed variable types."""

    def __init__(self):
        # Core components
        self.variables: Dict[str, z3.ExprRef] = {}  # name -> var
        self.prime_variables: Dict[str, z3.ExprRef] = {}  # name -> primed var
        self.invariants: List[z3.ExprRef] = []  # Multiple invariants
        self.init: z3.ExprRef = None  # Initial states
        self.trans: z3.ExprRef = None  # Transition relation
        self.post: z3.ExprRef = None  # Property to verify

        # Type tracking
        self.var_types: Dict[str, str] = {}  # Variable name -> type
        self.type_vars: Dict[str, Set[str]] = defaultdict(set)  # Type -> variable names

    def add_variable(
        self, name: str, sort: z3.SortRef, prime: bool = False
    ) -> z3.ExprRef:
        """Add a variable with given sort (type)"""
        var = z3.Const(name, sort)
        var_type = str(sort)

        if prime:
            self.prime_variables[name] = var
        else:
            self.variables[name] = var
            self.var_types[name] = var_type
            self.type_vars[var_type].add(name)

        return var

    def add_invariant(self, inv: z3.ExprRef) -> None:
        """Add an invariant"""
        self.invariants.append(inv)

    def set_init(self, init: z3.ExprRef) -> None:
        """Set initial states condition"""
        self.init = init

    def set_trans(self, trans: z3.ExprRef) -> None:
        """Set transition relation"""
        self.trans = trans

    def set_post(self, post: z3.ExprRef) -> None:
        """Set post-condition (property to verify)"""
        self.post = post

    def get_variables(self, var_type: Optional[str] = None) -> List[z3.ExprRef]:
        """Get all variables, optionally filtered by type"""
        if var_type is None:
            return list(self.variables.values())
        return [self.variables[name] for name in self.type_vars[var_type]]

    def get_prime_variables(self, var_type: Optional[str] = None) -> List[z3.ExprRef]:
        """Get all prime variables, optionally filtered by type"""
        if var_type is None:
            return list(self.prime_variables.values())
        return [self.prime_variables[name] for name in self.type_vars[var_type]]

    def get_all_variables(self) -> List[z3.ExprRef]:
        """Get all variables (both normal and prime)"""
        return self.get_variables() + self.get_prime_variables()

    def to_chc_constraints(self) -> z3.ExprRef:
        """Convert to CHC (Constrained Horn Clauses) constraints"""
        s = z3.SolverFor("HORN")

        # Create separate invariant functions for different variable types
        inv_funcs = {}
        for type_name, var_names in self.type_vars.items():
            if var_names:
                vars_of_type = [self.variables[name] for name in var_names]
                sort = vars_of_type[0].sort()
                inv_name = f"inv_{type_name}"
                inv_funcs[type_name] = z3.Function(
                    inv_name, *([sort] * len(vars_of_type)), z3.BoolSort()
                )

        # Initial states imply invariants
        for type_name, inv_func in inv_funcs.items():
            vars_of_type = [self.variables[name] for name in self.type_vars[type_name]]
            s.add(
                z3.ForAll(vars_of_type, z3.Implies(self.init, inv_func(*vars_of_type)))
            )

        # Invariants and transition relation imply next state invariants
        for type_name, inv_func in inv_funcs.items():
            curr_vars = [self.variables[name] for name in self.type_vars[type_name]]
            next_vars = [
                self.prime_variables[name] for name in self.type_vars[type_name]
            ]
            s.add(
                z3.ForAll(
                    self.get_all_variables(),
                    z3.Implies(
                        z3.And(inv_func(*curr_vars), self.trans), inv_func(*next_vars)
                    ),
                )
            )

        # Invariants imply post-condition
        for type_name, inv_func in inv_funcs.items():
            vars_of_type = [self.variables[name] for name in self.type_vars[type_name]]
            s.add(
                z3.ForAll(vars_of_type, z3.Implies(inv_func(*vars_of_type), self.post))
            )

        return z3.And(s.assertions())

    def to_chc_str(self) -> str:
        """Convert to CHC format string"""
        solver = z3.Solver()
        solver.add(self.to_chc_constraints())
        return "(set-logic HORN)\n" + solver.to_smt2()


def demo():
    """Demonstrate the NewTransitionSystem with a simple example."""
    # Create transition system
    ts = NewTransitionSystem()

    # Add variables: x (int) and y (bool)
    x = ts.add_variable("x", z3.IntSort())
    x_prime = ts.add_variable("x", z3.IntSort(), prime=True)
    y = ts.add_variable("y", z3.BoolSort())
    y_prime = ts.add_variable("y", z3.BoolSort(), prime=True)

    # Initial: x = 0, y = True
    ts.set_init(z3.And(x == 0, y == z3.BoolVal(True)))

    # Transition: x' = x + 1, y' = !y
    ts.set_trans(z3.And(x_prime == x + 1, y_prime == z3.Not(y)))

    # Property: x >= 0 (safety property)
    ts.set_post(x >= 0)

    # Convert to CHC and solve with PDR
    chc_constraints = ts.to_chc_constraints()
    solver = z3.SolverFor("HORN")
    solver.add(chc_constraints)

    print(chc_constraints)

    result = solver.check()
    print(f"PDR result: {result}")

    if result == z3.sat:
        print("Property holds!")
    elif result == z3.unsat:
        print("Property violated!")
    else:
        print("Unknown result")

    return result


if __name__ == "__main__":
    demo()
