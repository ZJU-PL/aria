from aria.tests import TestCase, main
from aria.itp.theories.logic.reverse_math_poc import (
    ExistsNat,
    ExistsSet,
    FiniteSecondOrderDecisionProcedure,
    ForallNat,
    ForallSet,
    Implies,
    InvalidFormulaError,
    NatConst,
    NatEq,
    NatIn,
    NatLt,
    NatAdd,
    NatVar,
    Not,
    free_variables,
    iff,
    parse_formula_text,
    parse_term_text,
)


class TestReverseMathPoc(TestCase):
    def test_bounded_comprehension_like_formula(self) -> None:
        proc = FiniteSecondOrderDecisionProcedure(universe_size=4)
        n = NatVar("n")

        # Exists X. Forall n<4. (n in X) <-> (n < 2)
        formula = ExistsSet(
            "X",
            ForallNat(
                "n",
                bound=4,
                body=iff(NatIn(n, "X"), NatLt(n, NatConst(2))),
            ),
        )

        result = proc.decide(formula)
        self.assertTrue(result.valid)
        self.assertGreater(result.evaluated_states, 0)

    def test_second_order_statement_not_valid(self) -> None:
        proc = FiniteSecondOrderDecisionProcedure(universe_size=3)
        n = NatVar("n")

        # Forall X. Exists n<3. not(n in X)
        # False in finite semantics because X can be the full universe.
        formula = ForallSet("X", ExistsNat("n", bound=3, body=Not(NatIn(n, "X"))))

        self.assertFalse(proc.decide(formula).valid)

    def test_simple_arithmetic_and_membership(self) -> None:
        proc = FiniteSecondOrderDecisionProcedure(universe_size=5)
        n = NatVar("n")

        # Exists X. Forall n<5. ((n in X) -> n = n)
        formula = ExistsSet(
            "X",
            ForallNat(
                "n",
                bound=5,
                body=Implies(NatIn(n, "X"), NatEq(n, n)),
            ),
        )

        self.assertTrue(proc.decide(formula).valid)

    def test_rejects_free_variables(self) -> None:
        proc = FiniteSecondOrderDecisionProcedure(universe_size=3)
        n = NatVar("n")
        with self.assertRaises(InvalidFormulaError):
            proc.decide(NatIn(n, "X"))

    def test_free_variables_utility(self) -> None:
        formula = ForallNat("n", 3, NatIn(NatVar("n"), "X"))
        free_nat, free_set = free_variables(formula)
        self.assertEqual(free_nat, set())
        self.assertEqual(free_set, {"X"})

    def test_safety_guard_for_universe_size(self) -> None:
        with self.assertRaises(ValueError):
            FiniteSecondOrderDecisionProcedure(universe_size=20)

    def test_parse_formula_and_decide(self) -> None:
        proc = FiniteSecondOrderDecisionProcedure(universe_size=4)
        formula = parse_formula_text(
            "(exists_set X (forall_nat n 4 (iff (in n X) (lt n 2))))"
        )
        self.assertTrue(proc.decide(formula).valid)

    def test_parse_term(self) -> None:
        term = parse_term_text("(+ 1 n)")
        self.assertIsInstance(term, NatAdd)
        self.assertEqual(term.left, NatConst(1))
        self.assertEqual(term.right, NatVar("n"))

    def test_parse_invalid_operator(self) -> None:
        with self.assertRaises(InvalidFormulaError):
            parse_formula_text("(foo a b)")


if __name__ == "__main__":
    main()
