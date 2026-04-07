import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import z3

from aria.automata.symautomata.pythondfa import PythonDFA
from aria.automata.symautomata.regex import Regex
from aria.automata.symautomata.sfa import SFA, SetPredicate, Z3Predicate


class TestSFA(unittest.TestCase):
    def test_set_predicate_boolean_operations(self):
        alphabet = [0, 1, 2, 3]
        left = SetPredicate([1, 2], alphabet)
        right = SetPredicate([2, 3], alphabet)

        self.assertEqual(set(left.conjunction(right)), {2})
        self.assertEqual(set(left.disjunction(right)), {1, 2, 3})
        self.assertEqual(set(left.negate()), {0, 3})

    def test_z3_predicate_membership_and_witness(self):
        symbol = z3.BitVec("sym", 8)
        universe = z3.ULT(symbol, z3.BitVecVal(4, 8))
        predicate = Z3Predicate(
            symbol,
            z3.ULT(symbol, z3.BitVecVal(3, 8)),
        )

        self.assertTrue(predicate.is_sat(universe=universe))
        self.assertTrue(predicate.is_sat(1, universe))
        self.assertFalse(predicate.is_sat(3, universe))

        witness = predicate.get_witness(universe)
        self.assertTrue(predicate.is_sat(witness, universe))

    def test_symbolic_acceptance_and_witness(self):
        symbol = z3.BitVec("sym", 8)
        automaton = SFA.symbolic(
            symbolic_symbol=symbol,
            symbolic_universe=z3.ULT(symbol, z3.BitVecVal(4, 8)),
            alphabet=[0, 1, 2, 3],
        )
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULT(symbol, z3.BitVecVal(2, 8)),
            ),
        )
        automaton.set_final(1, True)

        self.assertTrue(automaton.accepts([0]))
        self.assertTrue(automaton.accepts([1]))
        self.assertFalse(automaton.accepts([2]))

        witness = automaton.get_witness()
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertTrue(automaton.accepts(witness))

    def test_symbolic_bitvec_builder_sets_domain(self):
        automaton = SFA.symbolic_bitvec(
            8,
            alphabet=[0, 1, 2, 3],
            name="sym_builder",
            symbolic_universe=z3.ULT(z3.BitVec("sym_builder", 8), z3.BitVecVal(4, 8)),
        )
        assert automaton.symbolic_symbol is not None

        self.assertTrue(z3.is_bv(automaton.symbolic_symbol))
        self.assertEqual(automaton.symbolic_symbol.size(), 8)
        self.assertTrue(z3.is_true(z3.simplify(automaton.symbolic_symbol == z3.BitVec("sym_builder", 8))))
        self.assertTrue(
            z3.is_true(
                z3.simplify(
                    automaton.symbolic_universe
                    == z3.ULT(automaton.symbolic_symbol, z3.BitVecVal(4, 8))
                )
            )
        )

    def test_symbolic_int_builder_sets_domain(self):
        automaton = SFA.symbolic_int(
            alphabet=[0, 1, 2, 3],
            name="sym_int_builder",
            symbolic_universe=z3.And(
                z3.Int("sym_int_builder") >= 0,
                z3.Int("sym_int_builder") < 4,
            ),
        )
        assert automaton.symbolic_symbol is not None

        self.assertEqual(automaton.symbolic_symbol.sort_kind(), z3.Z3_INT_SORT)
        self.assertTrue(
            z3.is_true(
                z3.simplify(automaton.symbolic_symbol == z3.Int("sym_int_builder"))
            )
        )
        self.assertTrue(
            z3.is_true(
                z3.simplify(
                    automaton.symbolic_universe
                    == z3.And(
                        automaton.symbolic_symbol >= 0,
                        automaton.symbolic_symbol < 4,
                    )
                )
            )
        )

    def test_symbolic_bool_builder_sets_domain(self):
        automaton = SFA.symbolic_bool(
            alphabet=[False, True],
            name="sym_bool_builder",
        )
        assert automaton.symbolic_symbol is not None

        self.assertEqual(automaton.symbolic_symbol.sort_kind(), z3.Z3_BOOL_SORT)
        self.assertTrue(
            z3.is_true(
                z3.simplify(automaton.symbolic_symbol == z3.Bool("sym_bool_builder"))
            )
        )
        assert automaton.symbolic_universe is not None
        self.assertTrue(z3.is_true(z3.simplify(automaton.symbolic_universe)))

    def test_symbolic_int_language_operations(self):
        left = SFA.symbolic_int(
            name="sym_int_ops",
            symbolic_universe=z3.And(
                z3.Int("sym_int_ops") >= 0,
                z3.Int("sym_int_ops") < 4,
            ),
        )
        right = SFA.symbolic_int(
            name="sym_int_ops_other",
            symbolic_universe=z3.And(
                z3.Int("sym_int_ops_other") >= 0,
                z3.Int("sym_int_ops_other") < 4,
            ),
        )

        left_symbol = left.symbolic_symbol
        right_symbol = right.symbolic_symbol
        assert left_symbol is not None
        assert right_symbol is not None

        left.add_arc(0, 1, Z3Predicate(left_symbol, left_symbol <= 1))
        left.set_final(1, True)

        right.add_arc(0, 1, Z3Predicate(right_symbol, right_symbol >= 1))
        right.set_final(1, True)

        intersection = left.intersection(right)
        difference = left.difference(right)
        complement = left.complement()

        self.assertFalse(intersection.accepts([0]))
        self.assertTrue(intersection.accepts([1]))
        self.assertFalse(intersection.accepts([2]))

        self.assertTrue(difference.accepts([0]))
        self.assertFalse(difference.accepts([1]))
        self.assertFalse(difference.accepts([2]))

        self.assertFalse(complement.accepts([0]))
        self.assertFalse(complement.accepts([1]))
        self.assertTrue(complement.accepts([2]))
        self.assertFalse(complement.accepts([4]))

        equivalent = SFA.symbolic_int(
            name="sym_int_equiv",
            symbolic_universe=z3.And(
                z3.Int("sym_int_equiv") >= 0,
                z3.Int("sym_int_equiv") < 4,
            ),
        )
        equivalent_symbol = equivalent.symbolic_symbol
        assert equivalent_symbol is not None
        equivalent.add_arc(0, 1, Z3Predicate(equivalent_symbol, equivalent_symbol <= 1))
        equivalent.set_final(1, True)

        self.assertTrue(left.is_equivalent(equivalent))
        self.assertFalse(left.is_equivalent(right))

    def test_symbolic_bool_language_operations(self):
        left = SFA.symbolic_bool(name="sym_bool_ops")
        right = SFA.symbolic_bool(name="sym_bool_ops_other")

        left_symbol = left.symbolic_symbol
        right_symbol = right.symbolic_symbol
        assert left_symbol is not None
        assert right_symbol is not None

        left.add_arc(0, 1, Z3Predicate(left_symbol, left_symbol))
        left.set_final(1, True)

        right.add_arc(0, 1, Z3Predicate(right_symbol, z3.Not(right_symbol)))
        right.set_final(1, True)

        intersection = left.intersection(right)
        union = left.union(right)
        complement = left.complement()

        self.assertFalse(intersection.accepts([True]))
        self.assertFalse(intersection.accepts([False]))

        self.assertTrue(union.accepts([True]))
        self.assertTrue(union.accepts([False]))

        self.assertFalse(complement.accepts([True]))
        self.assertTrue(complement.accepts([False]))

        equivalent = SFA.symbolic_bool(name="sym_bool_equiv")
        equivalent_symbol = equivalent.symbolic_symbol
        assert equivalent_symbol is not None
        equivalent.add_arc(0, 1, Z3Predicate(equivalent_symbol, equivalent_symbol))
        equivalent.set_final(1, True)

        self.assertTrue(left.is_equivalent(equivalent))
        self.assertFalse(left.is_equivalent(right))

    def test_rejects_int_guard_on_bool_symbolic_automaton(self):
        automaton = SFA.symbolic_bool(name="sym_bool_validation")
        int_symbol = z3.Int("sym_int_validation")

        with self.assertRaises(TypeError):
            automaton.add_arc(0, 1, Z3Predicate(int_symbol, int_symbol >= 0))

    def test_rejects_bool_guard_on_int_symbolic_automaton(self):
        automaton = SFA.symbolic_int(name="sym_int_validation")
        bool_symbol = z3.Bool("sym_bool_validation")

        with self.assertRaises(TypeError):
            automaton.add_arc(0, 1, Z3Predicate(bool_symbol, bool_symbol))

    def test_z3_predicate_union_and_equivalence_across_universes(self):
        symbol = z3.BitVec("sym_union", 8)
        universe = z3.ULT(symbol, z3.BitVecVal(4, 8))
        low = Z3Predicate(
            symbol,
            z3.ULT(symbol, z3.BitVecVal(2, 8)),
        )
        high = Z3Predicate(
            symbol,
            z3.And(
                z3.UGE(symbol, z3.BitVecVal(2, 8)),
                z3.ULT(symbol, z3.BitVecVal(4, 8)),
            ),
        )
        whole = Z3Predicate(symbol, z3.ULT(symbol, z3.BitVecVal(4, 8)))

        combined = low.disjunction(high)

        self.assertTrue(combined.is_sat(0, universe))
        self.assertTrue(combined.is_sat(3, universe))
        self.assertFalse(combined.is_sat(4, universe))
        self.assertTrue(combined.is_equivalent(whole, universe))

    def test_boolean_language_operations(self):
        alphabet = [0, 1, 2, 3]
        left = SFA(alphabet)
        left.add_arc(0, 1, SetPredicate([1, 2], alphabet))
        left.set_final(1, True)

        right = SFA(alphabet)
        right.add_arc(0, 1, SetPredicate([2, 3], alphabet))
        right.set_final(1, True)

        self.assertTrue(left.intersection(right).accepts([2]))
        self.assertFalse(left.intersection(right).accepts([1]))

        self.assertTrue(left.union(right).accepts([3]))
        self.assertFalse(left.union(right).accepts([0]))

        self.assertTrue(left.difference(right).accepts([1]))
        self.assertFalse(left.difference(right).accepts([2]))

        self.assertTrue(left.complement().accepts([0]))
        self.assertFalse(left.complement().accepts([1]))

    def test_concretize_matches_finite_acceptance(self):
        alphabet = [0, 1, 2, 3]
        automaton = SFA(alphabet)
        automaton.add_arc(0, 1, SetPredicate([1, 2], alphabet))
        automaton.set_final(1, True)

        dfa = automaton.concretize()

        for symbol in alphabet:
            self.assertEqual(automaton.accepts([symbol]), dfa.consume_input([symbol]))

    def test_complement_requires_or_uses_predicate_factory(self):
        symbol = z3.BitVec("sym_factory", 8)
        incomplete = SFA()
        incomplete.add_state(initial=True, final=False)
        with self.assertRaises(ValueError):
            incomplete.complement()

        automaton = SFA(
            predicate_factory=lambda: Z3Predicate(
                symbol, z3.BoolVal(False)
            ),
            symbolic_symbol=symbol,
            symbolic_universe=z3.ULT(symbol, z3.BitVecVal(4, 8)),
        )
        automaton.add_state(initial=True, final=False)

        complement = automaton.complement()

        self.assertTrue(complement.accepts([]))
        self.assertTrue(complement.accepts([0]))
        self.assertTrue(complement.accepts([3]))

    def test_multistep_symbolic_path_and_roundtrip_load(self):
        symbol = z3.BitVec("sym_path", 8)
        automaton = SFA.symbolic(
            symbolic_symbol=symbol,
            symbolic_universe=z3.ULT(symbol, z3.BitVecVal(4, 8)),
            alphabet=[0, 1, 2, 3],
        )
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULT(symbol, z3.BitVecVal(2, 8)),
            ),
        )
        automaton.add_arc(
            1,
            2,
            Z3Predicate(
                symbol,
                z3.UGE(symbol, z3.BitVecVal(2, 8)),
            ),
        )
        automaton.set_final(2, True)

        self.assertTrue(automaton.accepts([1, 2]))
        self.assertFalse(automaton.accepts([1, 1]))

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "machine.txt"
            automaton.save(str(path))
            loaded = SFA([0, 1, 2, 3])
            loaded.load(str(path))

            self.assertTrue(loaded.accepts([1, 2]))
            self.assertFalse(loaded.accepts([1, 1]))

    def test_empty_automaton_boolean_operations(self):
        alphabet = [0, 1]
        empty = SFA(alphabet)
        base = SFA(alphabet)
        base.add_arc(0, 1, SetPredicate([1], alphabet))
        base.set_final(1, True)

        self.assertFalse(empty.accepts([]))
        self.assertTrue(empty.complement().accepts([]))
        self.assertTrue(empty.complement().accepts([0]))

        self.assertTrue(base.union(empty).accepts([1]))
        self.assertFalse(base.intersection(empty).accepts([1]))
        self.assertTrue(base.difference(empty).accepts([1]))
        self.assertTrue(base.is_equivalent(base.union(empty)))

    def test_from_acceptor_matches_dfa_language(self):
        dfa = PythonDFA(["a", "b"])
        dfa.add_arc(0, 1, "a")
        dfa.add_arc(1, 1, "b")
        dfa[1].final = True
        dfa[0].initial = True
        dfa.yy_accept = [1 if state.final else 0 for state in dfa.states]

        lifted = SFA.from_acceptor(dfa)

        self.assertTrue(lifted.accepts("a"))
        self.assertTrue(lifted.accepts("ab"))
        self.assertTrue(lifted.accepts("abb"))
        self.assertFalse(lifted.accepts(""))
        self.assertFalse(lifted.accepts("b"))

    def test_to_regex_uses_concrete_workflow(self):
        dfa = PythonDFA(["a", "b"])
        dfa.add_arc(0, 1, "a")
        dfa.add_arc(1, 1, "b")
        dfa[1].final = True
        dfa[0].initial = True
        dfa.yy_accept = [1 if state.final else 0 for state in dfa.states]

        lifted = SFA.from_acceptor(dfa)

        self.assertEqual(lifted.to_regex(), Regex(dfa).get_regex())

    def test_incremental_partition_preserves_overlapping_guards(self):
        symbol = z3.BitVec("sym_overlap", 8)
        automaton = SFA.symbolic(
            symbolic_symbol=symbol,
            symbolic_universe=z3.ULT(symbol, z3.BitVecVal(3, 8)),
            alphabet=[0, 1, 2, 3],
        )
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULE(symbol, z3.BitVecVal(1, 8)),
            ),
        )
        automaton.add_arc(
            0,
            2,
            Z3Predicate(
                symbol,
                z3.UGE(symbol, z3.BitVecVal(1, 8)),
            ),
        )
        automaton.set_final(1, True)
        automaton.set_final(2, True)

        deterministic = automaton.determinize()

        self.assertTrue(automaton.accepts([0]))
        self.assertTrue(automaton.accepts([1]))
        self.assertTrue(automaton.accepts([2]))
        self.assertFalse(automaton.accepts([3]))
        self.assertTrue(deterministic.accepts([0]))
        self.assertTrue(deterministic.accepts([1]))
        self.assertTrue(deterministic.accepts([2]))
        self.assertFalse(deterministic.accepts([3]))

    def test_determinize_uses_automaton_owned_symbolic_universe(self):
        symbol = z3.BitVec("sym_domain", 8)
        automaton = SFA.symbolic(
            symbolic_symbol=symbol,
            symbolic_universe=z3.ULT(symbol, z3.BitVecVal(4, 8)),
        )
        automaton.add_arc(
            0,
            1,
            Z3Predicate(symbol, z3.ULT(symbol, z3.BitVecVal(2, 8))),
        )
        automaton.add_arc(
            0,
            2,
            Z3Predicate(
                symbol,
                z3.And(
                    z3.UGE(symbol, z3.BitVecVal(2, 8)),
                    z3.ULT(symbol, z3.BitVecVal(4, 8)),
                ),
            ),
        )
        automaton.set_final(1, True)
        automaton.set_final(2, True)

        deterministic = automaton.determinize()

        for symbol_value in range(4):
            self.assertTrue(automaton.accepts([symbol_value]))
            self.assertTrue(deterministic.accepts([symbol_value]))
        self.assertFalse(deterministic.accepts([4]))

    def test_symbolic_union_rejects_mismatched_universes(self):
        left_symbol = z3.BitVec("sym_left", 8)
        left = SFA.symbolic(
            symbolic_symbol=left_symbol,
            symbolic_universe=z3.ULT(left_symbol, z3.BitVecVal(4, 8)),
        )
        left.add_arc(0, 1, Z3Predicate(left_symbol, z3.ULT(left_symbol, z3.BitVecVal(2, 8))))
        left.set_final(1, True)

        right_symbol = z3.BitVec("sym_right", 8)
        right = SFA.symbolic(
            symbolic_symbol=right_symbol,
            symbolic_universe=z3.ULT(right_symbol, z3.BitVecVal(5, 8)),
        )
        right.add_arc(
            0,
            1,
            Z3Predicate(right_symbol, z3.ULT(right_symbol, z3.BitVecVal(3, 8))),
        )
        right.set_final(1, True)

        with self.assertRaises(ValueError):
            left.union(right)

    def test_symbolic_language_operations_under_shared_universe(self):
        left = SFA.symbolic_bitvec(
            8,
            name="sym_ops",
            symbolic_universe=z3.ULT(z3.BitVec("sym_ops", 8), z3.BitVecVal(4, 8)),
        )
        right = SFA.symbolic_bitvec(
            8,
            name="sym_ops_other",
            symbolic_universe=z3.ULT(z3.BitVec("sym_ops_other", 8), z3.BitVecVal(4, 8)),
        )

        left_symbol = left.symbolic_symbol
        right_symbol = right.symbolic_symbol
        assert left_symbol is not None
        assert right_symbol is not None

        left.add_arc(0, 1, Z3Predicate(left_symbol, z3.ULT(left_symbol, z3.BitVecVal(2, 8))))
        left.set_final(1, True)

        right.add_arc(
            0,
            1,
            Z3Predicate(
                right_symbol,
                z3.And(
                    z3.UGE(right_symbol, z3.BitVecVal(1, 8)),
                    z3.ULT(right_symbol, z3.BitVecVal(4, 8)),
                ),
            ),
        )
        right.set_final(1, True)

        intersection = left.intersection(right)
        difference = left.difference(right)
        complement = left.complement()

        self.assertFalse(intersection.accepts([0]))
        self.assertTrue(intersection.accepts([1]))
        self.assertFalse(intersection.accepts([2]))

        self.assertTrue(difference.accepts([0]))
        self.assertFalse(difference.accepts([1]))
        self.assertFalse(difference.accepts([2]))

        self.assertFalse(complement.accepts([0]))
        self.assertFalse(complement.accepts([1]))
        self.assertTrue(complement.accepts([2]))
        self.assertFalse(complement.accepts([4]))

        equivalent = SFA.symbolic_bitvec(
            8,
            name="sym_equiv",
            symbolic_universe=z3.ULT(z3.BitVec("sym_equiv", 8), z3.BitVecVal(4, 8)),
        )
        equivalent_symbol = equivalent.symbolic_symbol
        assert equivalent_symbol is not None
        equivalent.add_arc(
            0,
            1,
            Z3Predicate(equivalent_symbol, z3.ULT(equivalent_symbol, z3.BitVecVal(2, 8))),
        )
        equivalent.set_final(1, True)

        self.assertTrue(left.is_equivalent(equivalent))
        self.assertFalse(left.is_equivalent(right))


if __name__ == "__main__":
    unittest.main()
