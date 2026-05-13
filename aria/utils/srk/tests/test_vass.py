"""
Tests for VASS closure construction.
"""

import unittest
from fractions import Fraction

from aria.utils.srk import smt
from aria.utils.srk import syntax
from aria.utils.srk import vass
from aria.utils.srk import vas
from aria.utils.srk.linalg.linear import QQMatrix, QQVector


class TestVASSClosure(unittest.TestCase):
    def setUp(self):
        self.srk = syntax.Context()
        self.x = syntax.mk_symbol(self.srk, "x", syntax.Type.INT)
        self.xp = syntax.mk_symbol(self.srk, "x_prime", syntax.Type.INT)
        self.k = syntax.mk_const(
            self.srk, syntax.mk_symbol(self.srk, "k", syntax.Type.INT)
        )
        self.tr_symbols = [(self.x, self.xp)]
        transformer = vas.Transformer(
            QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)})
        )
        self.scc = vass.SCCVAS(
            control_states=[syntax.mk_true(self.srk)],
            graph=[[vas.VAS.singleton(transformer)]],
            s_lst=[QQMatrix([QQVector({0: Fraction(1)})])],
        )

    def _check(self, formula, x_value, xp_value, k_value):
        constraints = syntax.mk_and(
            self.srk,
            [
                formula,
                syntax.mk_eq(
                    self.srk,
                    syntax.mk_const(self.srk, self.x),
                    syntax.mk_int(self.srk, x_value),
                ),
                syntax.mk_eq(
                    self.srk,
                    syntax.mk_const(self.srk, self.xp),
                    syntax.mk_int(self.srk, xp_value),
                ),
                syntax.mk_eq(self.srk, self.k, syntax.mk_int(self.srk, k_value)),
            ],
        )
        return smt.is_sat(self.srk, constraints)

    def test_closure_tracks_simple_scc_counter_balance(self):
        closure, sources = vass.closure_of_an_scc(
            self.srk, self.tr_symbols, self.k, self.scc
        )

        self.assertEqual(1, len(sources))
        self.assertEqual(smt.SMTResult.SAT, self._check(closure, 0, 3, 3))
        self.assertEqual(smt.SMTResult.UNSAT, self._check(closure, 0, 2, 3))

    def test_closure_requires_nonnegative_counters(self):
        closure, _ = vass.closure_of_an_scc(
            self.srk, self.tr_symbols, self.k, self.scc
        )

        self.assertEqual(smt.SMTResult.UNSAT, self._check(closure, -1, 0, 1))

    def test_exp_uses_scc_closure_not_placeholder(self):
        abstraction = vass.VASSType(
            vasses=[self.scc],
            formula=syntax.mk_true(self.srk),
            sink=syntax.mk_true(self.srk),
            skolem_constants=set(),
        )
        formula = vass.exp(self.srk, self.tr_symbols, self.k, abstraction)

        self.assertEqual(smt.SMTResult.SAT, self._check(formula, 0, 3, 3))
        self.assertEqual(smt.SMTResult.UNSAT, self._check(formula, 0, 2, 3))

    def test_closure_rejects_disconnected_positive_cycle_assignment(self):
        zero = vas.Transformer(
            QQVector({0: Fraction(1)}), QQVector({0: Fraction(0)})
        )
        increment = vas.Transformer(
            QQVector({0: Fraction(1)}), QQVector({0: Fraction(1)})
        )
        scc = vass.SCCVAS(
            control_states=[syntax.mk_true(self.srk), syntax.mk_true(self.srk)],
            graph=[
                [vas.VAS.singleton(zero), vas.VAS.empty()],
                [vas.VAS.empty(), vas.VAS.singleton(increment)],
            ],
            s_lst=[QQMatrix([QQVector({0: Fraction(1)})])],
        )

        closure, sources = vass.closure_of_an_scc(
            self.srk, self.tr_symbols, self.k, scc
        )
        force_first_source = syntax.mk_eq(
            self.srk, sources[0], syntax.mk_one(self.srk)
        )

        self.assertEqual(
            smt.SMTResult.UNSAT,
            self._check(
                syntax.mk_and(self.srk, [closure, force_first_source]),
                0,
                2,
                2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
