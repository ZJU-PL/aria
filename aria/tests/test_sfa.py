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
        predicate = Z3Predicate(
            symbol,
            z3.ULT(symbol, z3.BitVecVal(3, 8)),
            z3.ULT(symbol, z3.BitVecVal(4, 8)),
        )

        self.assertTrue(predicate.is_sat())
        self.assertTrue(predicate.is_sat(1))
        self.assertFalse(predicate.is_sat(3))

        witness = predicate.get_witness()
        self.assertTrue(predicate.is_sat(witness))

    def test_symbolic_acceptance_and_witness(self):
        symbol = z3.BitVec("sym", 8)
        automaton = SFA([0, 1, 2, 3])
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULT(symbol, z3.BitVecVal(2, 8)),
                z3.ULT(symbol, z3.BitVecVal(4, 8)),
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

    def test_z3_predicate_union_and_equivalence_across_universes(self):
        symbol = z3.BitVec("sym_union", 8)
        low = Z3Predicate(
            symbol,
            z3.BoolVal(True),
            z3.ULT(symbol, z3.BitVecVal(2, 8)),
        )
        high = Z3Predicate(
            symbol,
            z3.BoolVal(True),
            z3.And(
                z3.UGE(symbol, z3.BitVecVal(2, 8)),
                z3.ULT(symbol, z3.BitVecVal(4, 8)),
            ),
        )
        whole = Z3Predicate(symbol, z3.BoolVal(True), z3.ULT(symbol, z3.BitVecVal(4, 8)))

        combined = low.disjunction(high)

        self.assertTrue(combined.is_sat(0))
        self.assertTrue(combined.is_sat(3))
        self.assertFalse(combined.is_sat(4))
        self.assertTrue(combined.is_equivalent(whole))

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
                symbol, z3.BoolVal(False), z3.ULT(symbol, z3.BitVecVal(4, 8))
            )
        )
        automaton.add_state(initial=True, final=False)

        complement = automaton.complement()

        self.assertTrue(complement.accepts([]))
        self.assertTrue(complement.accepts([0]))
        self.assertTrue(complement.accepts([3]))

    def test_multistep_symbolic_path_and_roundtrip_load(self):
        symbol = z3.BitVec("sym_path", 8)
        automaton = SFA([0, 1, 2, 3])
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULT(symbol, z3.BitVecVal(2, 8)),
                z3.ULT(symbol, z3.BitVecVal(4, 8)),
            ),
        )
        automaton.add_arc(
            1,
            2,
            Z3Predicate(
                symbol,
                z3.UGE(symbol, z3.BitVecVal(2, 8)),
                z3.ULT(symbol, z3.BitVecVal(4, 8)),
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
        automaton = SFA([0, 1, 2, 3])
        automaton.add_arc(
            0,
            1,
            Z3Predicate(
                symbol,
                z3.ULE(symbol, z3.BitVecVal(1, 8)),
                z3.ULT(symbol, z3.BitVecVal(3, 8)),
            ),
        )
        automaton.add_arc(
            0,
            2,
            Z3Predicate(
                symbol,
                z3.UGE(symbol, z3.BitVecVal(1, 8)),
                z3.ULT(symbol, z3.BitVecVal(3, 8)),
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


if __name__ == "__main__":
    unittest.main()
