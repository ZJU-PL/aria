# Classes for linear and semi-linear sets

from aria.smt.lia_star.lia_star_solver import getModel
import aria.smt.lia_star.statistics
import itertools
import time
from z3 import *

# Check if V < U
def vecLess(V, U):
    return all([(0 <= v and v <= u) or (0 >= v and v >= u) for v, u in zip(V, U)])

# Subtract U from V
def vecSub(V, U):
    return [v - u for v, u in zip(V, U)]

# Class describing a linear set
class LS:

    # 'a' is the shift vector
    # 'B' is the set of basis vectors which can be linearly combined
    def __init__(self, a, B, phi):
        self.a = a
        self.B = B
        self.phi = phi

    # String rep
    def __repr__(self):
        return "{}".format([self.a, self.B])

    # Remove any duplicates from B
    def removeDuplicates(self):
        self.B.sort()
        self.B = list(b for b, _ in itertools.groupby(self.B))

    # lambda, lambda*B
    def linearCombination(self, name):

        # All zeroes returned in the case of empty basis
        if self.B == []:
            return ([], [0] * len(self.a))

        # Transpose the basis so that it is a list of rows, instead of a list of vectors:
        # [[x1, x2, ...], [y1, y2, ...]] -> [[x1, y1], [x2, y2], ...]
        transposed_basis = list(map(list, zip(*self.B)))

        # Make lambdas
        L = IntVector(name, len(self.B))

        # Linear combination with lambdas as coefficients
        LC = [Sum([l*v for v, l in list(zip(V, L))]) for V in transposed_basis]
        return L, LC

    # If possible without losing info, decreases the offset of a linear set
    def shiftDown(self):

        # Each b in B must be less than a to be considered
        a, B = self.a, self.B
        for b in B:
            if vecLess(b, a):

                # Solver and quantifiers
                s = Solver()
                L, LC = self.linearCombination('l1')
                non_negativity = [x >= 0 for x in L]

                # Assemble input
                input = [ai - bi + lci for (ai, bi, lci) in list(zip(a, b, LC))]

                # Check sat
                s.add(non_negativity + [Not(self.phi(input))])
                if None != getModel(s):
                   continue

                # Replace
                self.a = vecSub(a, b)
                aria.smt.lia_star.statistics.shiftdowns += 1
                return True
        return False

    # If possible without losing info, decreases a basis vector in a linear set
    def offsetDown(self):

        # Compare two b's in B, look for b2 <= b1
        a, B = self.a, self.B
        r = range(len(B))
        for i, j in itertools.product(r, r):
            if i == j: continue

            b1, b2 = B[i], B[j]
            if vecLess(b2, b1):

                # Basis to compare to
                B_new = list(B)
                B_new[i] = vecSub(b1, b2)
                new_set = LS(a, B_new, self.phi)

                # Solver and quantifiers
                s = Solver()
                L, LC = new_set.linearCombination('l1')
                non_negativity = [x >= 0 for x in L]

                # Assemble input
                input = [ai + lci for (ai, lci) in list(zip(a, LC))]

                # Check sat
                s.add(non_negativity + [Not(self.phi(input))])
                if None != getModel(s):
                   continue

                # Replace
                self.B = B_new
                aria.smt.lia_star.statistics.offsets += 1
                return True
        return False

    # Get the star of a single linear set and offset
    def star(self, mu, name):

        # Linear combination
        L, LC = self.linearCombination(name)

        # mu == 0 implies L == 0 and non-negativity
        fmls = [l >= 0 for l in L] + [mu >= 0]
        if L:
            fmls.append(Implies(mu == 0, And([l == 0 for l in L])))

        # Add offset vector to linear combination
        LC = [mu*ai + lci for (ai, lci) in list(zip(self.a, LC))]
        return L + [mu], LC, fmls

# Class describing a semi-linear set
class SLS:

    # 'sets' is a list of all linear sets in the sls.
    # 'phi' is the original LIA formula, a function that returns a Z3 expression
    # 'dim' is the number of args to phi
    def __init__(self, phi, set_vars, dimension):
        self.sets = [LS([0]*dimension, [], phi)]
        self.dim = dimension
        self.phi = phi
        self.set_vars = set_vars

    # Merges two compatible linear sets into one
    def _merge(self, i, j):
        if i == j:
            return False

        # a2 must be <= a1
        S1, S2 = self.sets[i], self.sets[j]
        a1, a2 = S1.a, S2.a
        if not vecLess(a2, a1):
            return False

        # Solver and quantifiers
        s = Solver()
        L1, LC1 = S1.linearCombination('l1')
        L2, LC2 = S2.linearCombination('l2')
        L3 = Int('l3')
        non_negativity = [x >= 0 for x in L1 + L2 + [L3]]

        # Assembling input to phi
        input = [a2i + lc1i + lc2i + L3*(a1i - a2i) for (a1i, a2i, lc1i, lc2i) in list(zip(a1, a2, LC1, LC2))]

        # Check sat
        s.add(non_negativity + [Not(self.phi(input))])
        if None != getModel(s):
           return False

        # Assemble new linear set and remove old ones
        new_set = LS(a2, S1.B + S2.B + [vecSub(a1, a2)], self.phi)
        del self.sets[max(i,j)]
        del self.sets[min(i,j)]
        self.sets.append(new_set)
        aria.smt.lia_star.statistics.merges += 1
        return True

    # Getter for the final semilinear set once the algorithm is done
    def getSLS(self):
        return self.sets

    # Number of vectors in the SLS
    def size(self):
        return sum([1 + len(ls.B) for ls in self.sets])

    # Let self.sets = { (a_1, B_1), ..., (a_n, B_n) }
    # Exists mu_i, lambda_i .
    #      X = Sum_i mu_i*a_i + lambda_i*B_i
    #      And_i mu_i >= 0 & lambda_i >= 0 & (mu_i = 0 => lambda_i = 0)
    # where mu_i, lambda_i are variables
    def starU(self, X=None):

        # Default args
        if not X:
            X = self.set_vars

        # Setup
        vars = []
        fmls = []
        sum = X
        mus = IntVector("mu", len(self.sets))

        # Accumulate sum for each set and add quantified variables as we go
        for i in range(len(self.sets)):

            # Get star of this set
            ls = self.sets[i]
            vs, s, fs = ls.star(mus[i], "l{}".format(i))

            # Cut linear combination to relevant projection
            s = s[:len(X)]

            # Assemble sum
            assert(len(sum) == len(s))
            sum = [sum[j] - s[j] for j in range(len(sum))]

            # Add variables
            vars += vs
            fmls += fs

        # Add summation
        fmls += [x == 0 for x in sum]
        return vars, fmls

    # Add existential quantifier to star so it can be safely used in other formulas
    def star(self, X=None):

        # Quantify an unquantified star
        vars, fmls = self.starU(X)
        return Exists(vars, And(fmls))

    # Attempt to apply merge, shiftDown, and offsetDown to reduce the size of the SLS
    def reduce(self):
        start = time.time()

        # Look for pairs of sets we can merge together
        done = False
        while not done:
            done = True
            idxs = range(len(self.sets))
            for i, j in itertools.product(idxs, idxs):
                if self._merge(i, j):
                    done = False
                    break

        # Try to decrease shifts and offsets
        for S in self.sets:
            while S.shiftDown(): pass
            while S.offsetDown(): pass
            S.removeDuplicates()

        end = time.time()
        aria.smt.lia_star.statistics.reduction_time += end - start

    # Add a new vector to the semi-linear set and return True
    # or return False if a vector cannot be added
    def augment(self):
        start = time.time()

        # Find non-negative X that satisfies phi and isn't reached by the current underapproximation
        s = Solver()
        X = IntVector('x', self.dim)
        s.add([x >= 0 for x in X])
        s.add(self.phi(X))
        s.add(Not(self.star(X)))

        # Get model and add new linear set to sls
        new_vec = getModel(s, X)
        end = time.time()
        aria.smt.lia_star.statistics.augment_time += end - start
        if new_vec != None:
            self.sets.append(LS(new_vec, [], self.phi))
            return True
        return False
