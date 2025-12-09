# Class describing an interpolant

from aria.smt.lia_star.lia_star_utils import getModel
import aria.smt.lia_star.statistics
import copy
from z3 import *


class Interpolant:

    # clauses is the list of interpolants for this problem
    # sls is the given semi-linear set
    # A is a function returning a Z3 expression
    # n is the number of args to A
    def __init__(self, A, B):
        self.clauses = []
        self.inductive_clauses = []
        self.sls = None
        self.A = A
        self.B = B

    # The sls underapproximation is updated with each iteration
    def update(self, sls):
        self.sls = sls

    # Getter function for the computed interpolants
    def getInductive(self):
        return self.inductive_clauses

    # Add an interpolant to the list if it isn't there already
    def _addClauses(self, new_i):

        # Break up conjunction into clauses if there is one
        new_clauses = new_i.children() if is_and(new_i) else [new_i]

        # For each clause, add if it's unique
        for nc in new_clauses:
            if not any([eq(nc, c) for c in self.clauses + self.inductive_clauses]):
                aria.smt.lia_star.statistics.interpolants_generated += 1
                self.clauses.append(nc)

    # Check if a given clause is inductive on the given set
    # (plus clauses which are already known to be inductive)
    def _checkInductive(self, clause, inductive_set):

        # Solver and vectors
        s = Solver()
        n = len(self.sls.set_vars)
        Y = IntVector('y', n)

        # Assert that Forall X, Y . I(X) ^ B(Y) => clause(X + Y)
        all_clauses = inductive_set + self.inductive_clauses
        non_negativity = [v >= 0 for v in self.sls.set_vars + Y]
        arg_sub = [(x, x + y) for (x, y) in list(zip(self.sls.set_vars, Y))]
        s.add(ForAll(self.B.args + Y, Implies(And(non_negativity + all_clauses + [self.B(Y)]), substitute(clause, arg_sub))))

        # Check satisfiability
        return getModel(s) != None

    # Calls spacer to get the interpolant between 'left' and 'right'
    def _interpolate(self, lvars, left, rvars, right, X, unfold, dir):

        # Create solver
        s = SolverFor('HORN')
        s.set("fp.xform.inline_eager", False)
        s.set("fp.xform.inline_linear", False)
        n = len(self.sls.set_vars)
        original = copy.copy(X)

        # Add the provided number of unfoldings to the interpolation problem
        if unfold > 0:

            # New input vector which sums X with the unfoldings
            Xx = IntVector("Xs", n)

            # Sum the unfoldings with X and add to left side
            sum, Xleft, fleft = self._getUnfoldings("Lx", unfold)
            unfoldFunc = (lambda a, b : a + b) if dir == "left" else (lambda a, b : a - b)
            left = And([left] + [fleft] + [Xx[i] == unfoldFunc(X[i], sum[i]) for i in range(n)])

            # Sum the unfoldings with X and add to right side
            sum, Xright, fright = self._getUnfoldings("Lx", unfold)
            unfoldFunc = (lambda a, b : a + b) if dir == "right" else (lambda a, b : a - b)
            right = And([right] + [fright] + [Xx[i] == unfoldFunc(X[i], sum[i]) for i in range(n)])

            # Add new variables to var list
            lvars += X + Xleft + [b for b in self.B.args if b not in self.sls.set_vars]
            rvars += X + Xright + [b for b in self.B.args if b not in self.sls.set_vars]

            # Set input vector to the new vector we created
            X = Xx

        # Left and right CHCs
        non_negativity_left = [x >= 0 for x in X + lvars]
        non_negativity_right = [x >= 0 for x in X + rvars]
        I = Function('I', [IntSort()] * n + [BoolSort()])
        s.add(ForAll(X + lvars, Implies(And(non_negativity_left + [left]), I(X))))
        s.add(ForAll(X + rvars, Implies(And([I(X)] + non_negativity_right + [right]), False)))

        # Check satisfiability (satisfiable inputs will sometimes fail to find an interpolant with unfoldings,
        # In this case the algorithm should terminate very shortly, so we just don't record an interpolant)
        aria.smt.lia_star.statistics.z3_calls += 1
        for i in range(50):
            if s.check() == sat:
                m = s.model()
                i = m.eval(I(original))
                return i
            elif s.check() == unsat:
                if unfold:
                    return None
                else:
                    print("error: interpolant.py: unsat interpolant")
                    exit(1)

        # If spacer wasn't able to compute an interpolant, then we can't add one on this iteration
        return None

    # Sum n vectors satisfying B together to get an unfolding of n steps,
    # to be added to the left and right side of an interpolation problem
    def _getUnfoldings(self, name, steps):

        n = len(self.sls.set_vars)

        # Each step adds a vector
        Xs = [IntVector('{}{}'.format(name, i), n) for i in range(steps)]

        # If there are no step vectors, their sum is 0
        if steps == 0:
            return [0]*n, [], True

        # Case for just one step
        if steps == 1:
            X_0 = Xs[0]
            fml = Or(And([x == 0 for x in X_0]), self.B(X_0))
            return X_0, X_0, fml

        # Case for many steps
        sum = [Sum([Xs[i][j] for i in range(steps)]) for j in range(n)]
        fml = True
        for i in range(steps):
            fml = Or(And([x == 0 for X in Xs[:i+1] for x in X]), And(self.B(Xs[i]), fml))
        return sum, [x for X in Xs for x in X], fml

    # Computes and records the forward interpolant for the given unfoldings
    def addForwardInterpolant(self, unfold=0):

        # Get B star and vars
        lambdas, star = self.sls.starU()

        # Interpolate and add result
        Avars = [a for a in self.A.args if a not in self.sls.set_vars]
        i = self._interpolate(lambdas, And(star), Avars, self.A(), self.sls.set_vars, unfold, "left")
        if i != None: self._addClauses(simplify(i))

    # Computes and records the backward interpolant for the given unfoldings
    def addBackwardInterpolant(self, unfold=0):

        # Get B star and vars
        lambdas, star = self.sls.starU()

        # Interpolate and add negated result
        Avars = [a for a in self.A.args if a not in self.sls.set_vars]
        i = self._interpolate(Avars, self.A(), lambdas, And(star), self.sls.set_vars, unfold, "right")
        if i != None: self._addClauses(simplify(Not(i)))

    # Filter all interpolants to only inductive clauses
    def filterToInductive(self):

        # Continue to apply the filter iteratively until every clause is kept
        inductive_subset = list(self.clauses)
        while True:

            # For each clause in the current set, keep if it's inductive on that set
            keep = []
            for c in inductive_subset:
                if self._checkInductive(c, inductive_subset):
                    keep.append(c)

            # Set the inductive interpolant to what was kept from the last iteration
            if inductive_subset == keep:
                break
            else:
                inductive_subset = list(keep)

        # Add inductive set to all known inductive clauses
        self.inductive_clauses += inductive_subset
