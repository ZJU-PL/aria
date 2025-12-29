"""Class describing an interpolant."""

import copy

from z3 import (
    And,
    BoolSort,
    ForAll,
    Function,
    Implies,
    IntSort,
    IntVector,
    Not,
    Or,
    Solver,
    SolverFor,
    Sum,
    substitute,
    simplify,
    is_and,
    eq,
    sat,
    unsat,
)

from aria.smt.lia_star.lia_star_utils import getModel
import aria.smt.lia_star.statistics


class Interpolant:
    """Class for computing and managing interpolants."""

    def __init__(self, a_func, b_func):
        """
        Initialize interpolant.

        Args:
            a_func: A function returning a Z3 expression
            b_func: A function returning a Z3 expression
        """
        self.clauses = []
        self.inductive_clauses = []
        self.sls = None
        self.a_func = a_func
        self.b_func = b_func

    def update(self, sls):
        """
        Update the sls underapproximation with each iteration.

        Args:
            sls: The semi-linear set
        """
        self.sls = sls

    def get_inductive(self):
        """
        Getter function for the computed interpolants.

        Returns:
            List of inductive clauses
        """
        return self.inductive_clauses

    def _add_clauses(self, new_i):
        """
        Add an interpolant to the list if it isn't there already.

        Args:
            new_i: New interpolant to add
        """
        # Break up conjunction into clauses if there is one
        new_clauses = new_i.children() if is_and(new_i) else [new_i]

        # For each clause, add if it's unique
        for nc in new_clauses:
            if not any(eq(nc, c) for c in self.clauses + self.inductive_clauses):
                aria.smt.lia_star.statistics.interpolants_generated += 1
                self.clauses.append(nc)

    def _check_inductive(self, clause, inductive_set):
        """
        Check if a given clause is inductive on the given set.

        Args:
            clause: Clause to check
            inductive_set: Set of clauses already known to be inductive

        Returns:
            True if clause is inductive, False otherwise
        """
        # Solver and vectors
        s = Solver()
        n = len(self.sls.set_vars)
        y_vars = IntVector('y', n)

        # Assert that Forall X, Y . I(X) ^ B(Y) => clause(X + Y)
        all_clauses = inductive_set + self.inductive_clauses
        non_negativity = [v >= 0 for v in self.sls.set_vars + y_vars]
        arg_sub = [(x, x + y) for (x, y) in list(zip(self.sls.set_vars, y_vars))]
        forall_expr = ForAll(
            self.b_func.args + y_vars,
            Implies(
                And(non_negativity + all_clauses + [self.b_func(y_vars)]),
                substitute(clause, arg_sub)
            )
        )
        s.add(forall_expr)

        # Check satisfiability
        return getModel(s) is not None

    def _interpolate(self, lvars, left, rvars, right, x_vars, unfold, direction):
        """
        Call spacer to get the interpolant between 'left' and 'right'.

        Args:
            lvars: Left variables
            left: Left formula
            rvars: Right variables
            right: Right formula
            x_vars: Input vector
            unfold: Number of unfoldings
            direction: Direction of unfolding ("left" or "right")

        Returns:
            Interpolant or None
        """
        # Create solver
        s = SolverFor('HORN')
        s.set("fp.xform.inline_eager", False)
        s.set("fp.xform.inline_linear", False)
        n = len(self.sls.set_vars)
        original = copy.copy(x_vars)

        # Add the provided number of unfoldings to the interpolation problem
        if unfold > 0:

            # New input vector which sums x_vars with the unfoldings
            xx_vars = IntVector("Xs", n)

            # Sum the unfoldings with x_vars and add to left side
            sum_left, xleft, fleft = self._get_unfoldings("Lx", unfold)
            unfold_func = (lambda a, b: a + b) if direction == "left" else (lambda a, b: a - b)
            left = And([left] + [fleft] + [xx_vars[i] == unfold_func(x_vars[i], sum_left[i]) for i in range(n)])

            # Sum the unfoldings with x_vars and add to right side
            sum_right, xright, fright = self._get_unfoldings("Lx", unfold)
            unfold_func = (lambda a, b: a + b) if direction == "right" else (lambda a, b: a - b)
            right = And([right] + [fright] + [xx_vars[i] == unfold_func(x_vars[i], sum_right[i]) for i in range(n)])

            # Add new variables to var list
            lvars += x_vars + xleft + [b for b in self.b_func.args if b not in self.sls.set_vars]
            rvars += x_vars + xright + [b for b in self.b_func.args if b not in self.sls.set_vars]

            # Set input vector to the new vector we created
            x_vars = xx_vars

        # Left and right CHCs
        non_negativity_left = [x >= 0 for x in x_vars + lvars]
        non_negativity_right = [x >= 0 for x in x_vars + rvars]
        i_func = Function('I', [IntSort()] * n + [BoolSort()])
        s.add(ForAll(x_vars + lvars, Implies(And(non_negativity_left + [left]), i_func(x_vars))))
        s.add(ForAll(x_vars + rvars, Implies(And([i_func(x_vars)] + non_negativity_right + [right]), False)))

        # Check satisfiability (satisfiable inputs will sometimes fail to find
        # an interpolant with unfoldings. In this case the algorithm should
        # terminate very shortly, so we just don't record an interpolant)
        aria.smt.lia_star.statistics.z3_calls += 1
        for _ in range(50):
            if s.check() == sat:
                m = s.model()
                interp = m.eval(i_func(original))
                return interp
            if s.check() == unsat:
                if unfold:
                    return None
                print("error: interpolant.py: unsat interpolant")
                exit(1)

        # If spacer wasn't able to compute an interpolant, then we can't add
        # one on this iteration
        return None

    def _get_unfoldings(self, name, steps):
        """
        Sum n vectors satisfying B together to get an unfolding of n steps.

        To be added to the left and right side of an interpolation problem.

        Args:
            name: Base name for variables
            steps: Number of steps

        Returns:
            Tuple of (sum, variables, formula)
        """
        n = len(self.sls.set_vars)

        # Each step adds a vector
        xs_vars = [IntVector(f'{name}{i}', n) for i in range(steps)]

        # If there are no step vectors, their sum is 0
        if steps == 0:
            return [0]*n, [], True

        # Case for just one step
        if steps == 1:
            x_0 = xs_vars[0]
            fml = Or(And([x == 0 for x in x_0]), self.b_func(x_0))
            return x_0, x_0, fml

        # Case for many steps
        sum_vars = [Sum([xs_vars[i][j] for i in range(steps)]) for j in range(n)]
        fml = True
        for i in range(steps):
            fml = Or(And([x == 0 for x_var in xs_vars[:i+1] for x in x_var]), And(self.b_func(xs_vars[i]), fml))
        return sum_vars, [x for x_var in xs_vars for x in x_var], fml

    def add_forward_interpolant(self, unfold=0):
        """
        Compute and record the forward interpolant for the given unfoldings.

        Args:
            unfold: Number of unfoldings (default: 0)
        """
        # Get B star and vars
        lambdas, star = self.sls.starU()

        # Interpolate and add result
        avars = [a for a in self.a_func.args if a not in self.sls.set_vars]
        interp = self._interpolate(
            lambdas, And(star), avars, self.a_func(), self.sls.set_vars, unfold, "left"
        )
        if interp is not None:
            self._add_clauses(simplify(interp))

    def add_backward_interpolant(self, unfold=0):
        """
        Compute and record the backward interpolant for the given unfoldings.

        Args:
            unfold: Number of unfoldings (default: 0)
        """
        # Get B star and vars
        lambdas, star = self.sls.starU()

        # Interpolate and add negated result
        avars = [a for a in self.a_func.args if a not in self.sls.set_vars]
        interp = self._interpolate(
            avars, self.a_func(), lambdas, And(star), self.sls.set_vars, unfold, "right"
        )
        if interp is not None:
            self._add_clauses(simplify(Not(interp)))

    def filter_to_inductive(self):
        """Filter all interpolants to only inductive clauses."""
        # Continue to apply the filter iteratively until every clause is kept
        inductive_subset = list(self.clauses)
        while True:

            # For each clause in the current set, keep if it's inductive on that set
            keep = []
            for c in inductive_subset:
                if self._check_inductive(c, inductive_subset):
                    keep.append(c)

            # Set the inductive interpolant to what was kept from the last iteration
            if inductive_subset == keep:
                break
            inductive_subset = list(keep)

        # Add inductive set to all known inductive clauses
        self.inductive_clauses += inductive_subset
