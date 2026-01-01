# Classes for linear and semi-linear sets

import itertools
import time

from z3 import And, Exists, Implies, Int, IntVector, Not, Solver, Sum

from aria.smt.lia_star.lia_star_utils import getModel
import aria.smt.lia_star.statistics


# Check if V < U
def vec_less(vec_v, vec_u):  # pylint: disable=invalid-name
    """Check if vec_v < vec_u component-wise."""
    return all((0 <= v <= u) or (0 >= v >= u) for v, u in zip(vec_v, vec_u))


# Subtract U from V
def vec_sub(vec_v, vec_u):  # pylint: disable=invalid-name
    """Subtract vec_u from vec_v component-wise."""
    return [v - u for v, u in zip(vec_v, vec_u)]


# Class describing a linear set
class LS:  # pylint: disable=invalid-name

    # 'a' is the shift vector
    # 'basis' is the set of basis vectors which can be linearly combined
    def __init__(self, a, basis, phi):  # pylint: disable=invalid-name
        self.a = a
        self.B = basis  # pylint: disable=invalid-name
        self.phi = phi

    # String rep
    def __repr__(self):
        return f"{[self.a, self.B]}"

    # Remove any duplicates from B
    def remove_duplicates(self):  # pylint: disable=invalid-name
        """Remove duplicate basis vectors."""
        self.B.sort()
        self.B = [b for b, _ in itertools.groupby(self.B)]

    # lambda, lambda*B
    def linear_combination(self, name):  # pylint: disable=invalid-name
        """Compute linear combination lambda*B."""
        # All zeroes returned in the case of empty basis
        if not self.B:
            return ([], [0] * len(self.a))

        # Transpose the basis so that it is a list of rows, instead of a list of vectors:
        # [[x1, x2, ...], [y1, y2, ...]] -> [[x1, y1], [x2, y2], ...]
        transposed_basis = list(map(list, zip(*self.B)))

        # Make lambdas
        lambdas = IntVector(name, len(self.B))  # pylint: disable=invalid-name

        # Linear combination with lambdas as coefficients
        linear_combo = [  # pylint: disable=invalid-name
            Sum([l * v for v, l in zip(row, lambdas)]) for row in transposed_basis
        ]
        return lambdas, linear_combo

    # If possible without losing info, decreases the offset of a linear set
    def shift_down(self):  # pylint: disable=invalid-name
        """Decrease the offset of a linear set if possible."""
        # Each b in B must be less than a to be considered
        a = self.a
        basis = self.B
        for b in basis:
            if vec_less(b, a):

                # Solver and quantifiers
                s = Solver()
                lambdas, linear_combo = self.linear_combination("l1")
                non_negativity = [x >= 0 for x in lambdas]

                # Assemble input
                input_vec = [  # pylint: disable=invalid-name
                    ai - bi + lci for (ai, bi, lci) in zip(a, b, linear_combo)
                ]

                # Check sat
                s.add(non_negativity + [Not(self.phi(input_vec))])
                if getModel(s) is not None:
                    continue

                # Replace
                self.a = vec_sub(a, b)
                aria.smt.lia_star.statistics.shiftdowns += 1
                return True
        return False

    # If possible without losing info, decreases a basis vector in a linear set
    def offset_down(self):  # pylint: disable=invalid-name
        """Decrease a basis vector in a linear set if possible."""
        # Compare two b's in B, look for b2 <= b1
        a = self.a
        basis = self.B
        r = range(len(basis))
        for i, j in itertools.product(r, r):
            if i == j:
                continue

            b1, b2 = basis[i], basis[j]
            if vec_less(b2, b1):

                # Basis to compare to
                basis_new = list(basis)  # pylint: disable=invalid-name
                basis_new[i] = vec_sub(b1, b2)
                new_set = LS(a, basis_new, self.phi)

                # Solver and quantifiers
                s = Solver()
                lambdas, linear_combo = new_set.linear_combination("l1")
                non_negativity = [x >= 0 for x in lambdas]

                # Assemble input
                input_vec = [  # pylint: disable=invalid-name
                    ai + lci for (ai, lci) in zip(a, linear_combo)
                ]

                # Check sat
                s.add(non_negativity + [Not(self.phi(input_vec))])
                if getModel(s) is not None:
                    continue

                # Replace
                self.B = basis_new
                aria.smt.lia_star.statistics.offsets += 1
                return True
        return False

    # Get the star of a single linear set and offset
    def star(self, mu, name):
        """Get the star of a single linear set."""
        # Linear combination
        lambdas, linear_combo = self.linear_combination(name)

        # mu == 0 implies L == 0 and non-negativity
        fmls = [l >= 0 for l in lambdas] + [mu >= 0]
        if lambdas:
            fmls.append(Implies(mu == 0, And([l == 0 for l in lambdas])))

        # Add offset vector to linear combination
        linear_combo = [mu * ai + lci for (ai, lci) in zip(self.a, linear_combo)]
        return lambdas + [mu], linear_combo, fmls


# Class describing a semi-linear set
class SLS:  # pylint: disable=invalid-name

    # 'sets' is a list of all linear sets in the sls.
    # 'phi' is the original LIA formula, a function that returns a Z3 expression
    # 'dim' is the number of args to phi
    def __init__(self, phi, set_vars, dimension):
        self.sets = [LS([0] * dimension, [], phi)]
        self.dim = dimension
        self.phi = phi
        self.set_vars = set_vars

    # Merges two compatible linear sets into one
    def _merge(self, i, j):
        """Merge two compatible linear sets into one."""
        if i == j:
            return False

        # a2 must be <= a1
        set1, set2 = self.sets[i], self.sets[j]  # pylint: disable=invalid-name
        a1, a2 = set1.a, set2.a  # pylint: disable=invalid-name
        if not vec_less(a2, a1):
            return False

        # Solver and quantifiers
        s = Solver()
        lambdas1, linear_combo1 = set1.linear_combination(
            "l1"
        )  # pylint: disable=invalid-name
        lambdas2, linear_combo2 = set2.linear_combination(
            "l2"
        )  # pylint: disable=invalid-name
        lambda3 = Int("l3")  # pylint: disable=invalid-name
        non_negativity = [x >= 0 for x in lambdas1 + lambdas2 + [lambda3]]

        # Assembling input to phi
        input_vec = [  # pylint: disable=invalid-name
            a2i + lc1i + lc2i + lambda3 * (a1i - a2i)
            for (a1i, a2i, lc1i, lc2i) in zip(a1, a2, linear_combo1, linear_combo2)
        ]

        # Check sat
        s.add(non_negativity + [Not(self.phi(input_vec))])
        if getModel(s) is not None:
            return False

        # Assemble new linear set and remove old ones
        new_set = LS(a2, set1.B + set2.B + [vec_sub(a1, a2)], self.phi)
        del self.sets[max(i, j)]
        del self.sets[min(i, j)]
        self.sets.append(new_set)
        aria.smt.lia_star.statistics.merges += 1
        return True

    # Getter for the final semilinear set once the algorithm is done
    def get_sls(self):  # pylint: disable=invalid-name
        """Get the semilinear set."""
        return self.sets

    # Number of vectors in the SLS
    def size(self):
        """Get the size of the SLS."""
        return sum(1 + len(ls.B) for ls in self.sets)

    # Let self.sets = { (a_1, B_1), ..., (a_n, B_n) }
    # Exists mu_i, lambda_i .
    #      X = Sum_i mu_i*a_i + lambda_i*B_i
    #      And_i mu_i >= 0 & lambda_i >= 0 & (mu_i = 0 => lambda_i = 0)
    # where mu_i, lambda_i are variables
    def star_u(self, x_vars=None):  # pylint: disable=invalid-name
        """Get unquantified star representation."""
        # Default args
        if not x_vars:
            x_vars = self.set_vars

        # Setup
        var_list = []
        fmls = []
        sum_vec = x_vars  # pylint: disable=invalid-name
        mus = IntVector("mu", len(self.sets))

        # Accumulate sum for each set and add quantified variables as we go
        for i, ls in enumerate(self.sets):

            # Get star of this set
            vs, s, fs = ls.star(mus[i], f"l{i}")

            # Cut linear combination to relevant projection
            s = s[: len(x_vars)]

            # Assemble sum
            assert len(sum_vec) == len(s)
            sum_vec = [sum_vec[j] - s[j] for j in range(len(sum_vec))]

            # Add variables
            var_list += vs
            fmls += fs

        # Add summation
        fmls += [x == 0 for x in sum_vec]
        return var_list, fmls

    # Add existential quantifier to star so it can be safely used in other formulas
    def star(self, x_vars=None):
        """Get quantified star representation."""
        # Quantify an unquantified star
        var_list, fmls = self.star_u(x_vars)
        return Exists(var_list, And(fmls))

    # Attempt to apply merge, shiftDown, and offsetDown to reduce the size of the SLS
    def reduce(self):
        """Reduce the size of the SLS."""
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
        for linear_set in self.sets:
            while linear_set.shift_down():
                pass
            while linear_set.offset_down():
                pass
            linear_set.remove_duplicates()

        end = time.time()
        aria.smt.lia_star.statistics.reduction_time += end - start

    # Add a new vector to the semi-linear set and return True
    # or return False if a vector cannot be added
    def augment(self):
        """Add a new vector to the semi-linear set."""
        start = time.time()

        # Find non-negative X that satisfies phi and isn't reached by the current underapproximation
        s = Solver()
        x_vec = IntVector("x", self.dim)  # pylint: disable=invalid-name
        s.add([x >= 0 for x in x_vec])
        s.add(self.phi(x_vec))
        s.add(Not(self.star(x_vec)))

        # Get model and add new linear set to sls
        new_vec = getModel(s, x_vec)
        end = time.time()
        aria.smt.lia_star.statistics.augment_time += end - start
        if new_vec is not None:
            self.sets.append(LS(new_vec, [], self.phi))
            return True
        return False
