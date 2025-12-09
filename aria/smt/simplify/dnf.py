# Dumb dnf converter

from pysmt.simplifier import BddSimplifier
from pysmt.walkers import IdentityDagWalker
import pysmt.operators as operators

from pysmt.shortcuts import (
    Real,
    TRUE, FALSE,
    Minus,
    LE, LT,
    Not, And, Or
)

class BddDNFSimplifier(BddSimplifier):
    def __init__(self, env=None, static_ordering=None, bool_abstraction=True):
        BddSimplifier.__init__(self, env=env,
                               static_ordering=static_ordering,
                               bool_abstraction=True)

        self.dnf = True

    def abstract_and_simplify(self, formula):
        abs_formula = self.walk(formula)
        bdd_formula = self.convert(abs_formula)

        try:
            import repycudd

            if (self.dnf):
                # build DNF using cubes
                m = self.s.ddmanager
                repycudd.set_iter_meth(0)

                abs_res = FALSE()
                for cube in repycudd.ForeachCubeIterator(m, bdd_formula):
                    bdd_cube = TRUE()

                    i = -1
                    for cube in cube:
                        i = i + 1
                        var = self.back(m.IthVar(i))

                        if cube == 1:
                            bdd_cube = And(bdd_cube, var)
                        elif cube == 0:
                            bdd_cube = And(bdd_cube, Not(var))
                    abs_res = Or(abs_res, bdd_cube)
            else:
                abs_res = self.back(bdd_formula)

            res = abs_res.substitute(self.ba_map)
            return res

        except:
            raise Exception("Cannot load BDD package")

class DNFConverter(object):

    class PostProcess(IdentityDagWalker):
        """
        Remove the negation in front of the predicates changing the
        polynomial.

        It assumes the input formula is in NNF
        """
        def __init__(self, env=None, invalidate_memoization=None):
            IdentityDagWalker.__init__(self,
                                       env=env,
                                       invalidate_memoization=invalidate_memoization)
            self.mgr = self.env.formula_manager

        def walk_not(self, formula, args, **kwargs):
            # Assume we are in NNF, so we have atoms
            assert(len(args) == 1)

            predicate = args[0]
            if (predicate.is_le()):
                # (! p <= 0) <-> p > 0 <-> -p < 0
                return LT(Minus(predicate.args()[1], predicate.args()[0]), Real(0))
            elif (predicate.is_lt()):
                # (! p < 0) <-> p >= 0 <-> -p <= 0
                return LE(Minus(predicate.args()[1], predicate.args()[0]), Real(0))
                pass
            else:
                raise Exception("Not supported")

    class PreProcess(IdentityDagWalker):
        """
        Pre-process a formula removing the = operators.

        This to ensure we only have predicates using >=, >
        """
        def __init__(self, env=None, invalidate_memoization=None):
            IdentityDagWalker.__init__(self,
                                       env=env,
                                       invalidate_memoization=invalidate_memoization)
            self.mgr = self.env.formula_manager

        def walk_equals(self, formula, args, **kwargs):
            # p1 = p2 <-> p1 - p2 = 0 <-> (p1 - p2 <= 0) and (0 >= p1 - p2)
            predicate = Minus(args[0], args[1])
            return And(LE(predicate, Real(0)), LE(Real(0), predicate))


    def __init__(self, env = None):
        self.env = env

    def get_dnf(self, formula):
        """
        Returns a DNF representation of formula
        """
        try:
            import repycudd
        except:
            raise Exception("Cannot load BDD Package")

        # Very expensive DNF-ization, enumerates all the models
        # Equivalent DNF is the one ok for LZZ
        s = BddDNFSimplifier(env=self.env, bool_abstraction=True)

        pre_processor = DNFConverter.PreProcess(env = self.env)
        post_processor = DNFConverter.PostProcess(env = self.env)

        pre_processed = pre_processor.walk(formula)
        simplified_formula = s.simplify(pre_processed)
        post_processed = post_processor.walk(simplified_formula)

        return simplified_formula



class ApplyPredicate(IdentityDagWalker):
    """
    Apply a function map_f[predicate] to each formula
    containing predicate.

    Note: the walker is restricted to predicates (<, <=, =)
    """
    def __init__(self, map_f, env=None, invalidate_memoization=None):
        IdentityDagWalker.__init__(self,
                                   env=env,
                                   invalidate_memoization=invalidate_memoization)
        self.mgr = self.env.formula_manager
        # f should be a callable function, it could be partial
        self.map_f = map_f

    def walk_lt(self, formula, args, **kwargs):
        predicate = Minus(args[0], args[1])
        f = self.map_f[operators.LT]
        new_node = f(predicate)
        return new_node

    def walk_le(self, formula, args, **kwargs):
        predicate = Minus(args[0], args[1])
        f = self.map_f[operators.LE]
        new_node = f(predicate)

        return new_node

    def walk_equals(self, formula, args, **kwargs):
        predicate = Minus(args[0], args[1])
        f = self.map_f[operators.EQUALS]
        new_node = f(predicate)
        return new_node
