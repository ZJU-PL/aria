from aria.cflobdd import (
    GrammarProduction,
    GrammarReachability,
    Relation,
    balanced_reachability,
    build_witness_tree,
    extract_edge_path,
    reflexive_transitive_closure,
)
from aria.tests import TestCase, main


class TestCFLOBDDReachability(TestCase):
    def test_compose_with_existential_elimination(self):
        left = Relation.from_edges([(0, 1), (1, 2)], name="left")
        right = Relation.from_edges([(1, 3), (2, 4)], name="right")

        composed = left.binary_compose(right, name="composed")

        self.assertEqual(composed.tuples(), [(0, 3), (1, 4)])
        witness = composed.witness({"src": 0, "dst": 3})
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness["kind"], "quantified_compose")
        self.assertEqual(witness["eliminate"], ["mid"])
        self.assertEqual(extract_edge_path(witness), [(0, 1), (1, 3)])

    def test_symbolic_relation_and_labeled_quantified_composition(self):
        relation = Relation.from_labeled_edges([(0, 7, 1), (1, 8, 2)], name="labels")
        self.assertIsNotNone(relation.block_encoding())
        self.assertIsNotNone(relation.symbolic)
        self.assertGreater(relation.symbolic_solutions(), 0)

        left = Relation.from_tuples(
            ("src", "label", "mid"),
            [(0, 7, 1), (1, 8, 2)],
            name="left_labeled",
        )
        right = Relation.from_tuples(
            ("mid", "weight", "dst"),
            [(1, 5, 9), (2, 6, 10)],
            name="right_labeled",
        )

        composed = left.quantified_compose(
            right,
            shared=("mid",),
            eliminate=("mid",),
            keep=("src", "label", "weight", "dst"),
            name="joined",
        )

        self.assertEqual(composed.tuples(), [(0, 7, 5, 9), (1, 8, 6, 10)])
        self.assertIsNotNone(composed.symbolic)

    def test_reflexive_transitive_closure(self):
        relation = Relation.from_edges([(0, 1), (1, 2), (2, 3)], name="chain")

        closure = reflexive_transitive_closure(
            relation, nodes=[0, 1, 2, 3], max_iterations=10
        )

        self.assertTrue(closure.converged)
        self.assertTrue(closure.relation.contains({"src": 0, "dst": 3}))
        self.assertTrue(closure.relation.contains({"src": 2, "dst": 2}))

    def test_grammar_nonterminal_solving(self):
        grammar = GrammarReachability(
            start_symbol="S",
            productions=[
                GrammarProduction("S", ("A", "B")),
                GrammarProduction("A", ("a",)),
                GrammarProduction("B", ("b",)),
                GrammarProduction("I", ()),
                GrammarProduction("T", ("a", "b", "c")),
            ],
        )

        solution = grammar.solve(
            terminals={
                "a": Relation.from_edges([(0, 1)], name="a"),
                "b": Relation.from_edges([(1, 2)], name="b"),
                "c": Relation.from_edges([(2, 3)], name="c"),
            },
            nodes=[0, 1, 2, 3],
            max_iterations=10,
        )

        self.assertTrue(solution.converged)
        self.assertTrue(solution.contains("S", 0, 2))
        self.assertTrue(solution.contains("I", 1, 1))
        witness = solution.witness("S", 0, 2)
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness["kind"], "grammar")
        self.assertEqual(witness["symbol"], "S")
        self.assertEqual(extract_edge_path(witness), [(0, 1), (1, 2)])
        self.assertTrue(solution.contains("T", 0, 3))
        tree = build_witness_tree(solution.witness("T", 0, 3))
        self.assertEqual(tree["kind"], "grammar")
        self.assertGreaterEqual(len(tree["children"]), 1)

    def test_balanced_call_return_reachability(self):
        intra = Relation.from_edges([(1, 2)], label="intra", name="intra")
        calls = {"c": Relation.from_edges([(0, 1)], label="call", name="call")}
        returns = {
            "c": Relation.from_edges([(2, 3)], label="return", name="return")
        }

        result = balanced_reachability(
            intra,
            calls,
            returns,
            nodes=[0, 1, 2, 3],
            include_identity=False,
            max_iterations=10,
        )

        self.assertTrue(result.converged)
        self.assertTrue(result.relation.contains({"src": 0, "dst": 3}))
        witness = result.relation.witness({"src": 0, "dst": 3})
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertEqual(witness["kind"], "balanced")
        self.assertEqual(witness["rule"], "call_return")
        self.assertEqual(extract_edge_path(witness), [(0, 1), (1, 2), (2, 3)])


if __name__ == "__main__":
    main()
