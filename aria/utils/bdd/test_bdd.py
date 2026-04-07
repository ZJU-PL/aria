import unittest

from aria.utils.bdd import BDDNode, OBDD


class TestOBDD(unittest.TestCase):
    def setUp(self):
        self.a = BDDNode("a", BDDNode(0), BDDNode(True))
        self.b = BDDNode("a", BDDNode(True), BDDNode(False))
        self.c = BDDNode("c", self.b, BDDNode(False))
        self.d = BDDNode("b", BDDNode(1), BDDNode(0))

        self.ordering = ["c", "a"]

    def test_bdd(self):
        for bdd, text, bdd_vars in [
            (self.a, "a", ["a"]),
            (self.b, "~a", ["a"]),
            (self.c, "~c & ~a", ["a", "c"]),
            (self.d, "~b", ["b"]),
        ]:

            self.assertEqual(f"{bdd}", text)
            self.assertEqual(bdd.variables(), set(bdd_vars))

        def get_ancestors():
            return [f"{bdd_node}" for bdd_node in BDDNode.nodes()]

        before_e = get_ancestors()

        e = BDDNode("r", BDDNode("c", self.d, self.a), BDDNode("h", self.a, self.d))

        with_e = get_ancestors()

        e = None

        after_e = get_ancestors()

        self.assertEqual(before_e, after_e)
        self.assertNotEqual(with_e, after_e)

    def test_obdd(self):
        oa = OBDD(self.a, self.ordering)
        ob = OBDD(self.c, self.ordering)

        tests_list = [
            (oa, "lambda c,a: a", ["a"]),
            (ob, "lambda c,a: ~c & ~a", ["a", "c"]),
            (ob.restrict("c", False), "lambda c,a: ~a", ["a"]),
            (ob.restrict("c", 1), "lambda c,a: 0", []),
            (ob.restrict("a", 0), "lambda c,a: ~c", ["c"]),
            (ob.restrict("a", True), "lambda c,a: 0", []),
            (~ob, "lambda c,a: (~c & a) | (c)", ["a", "c"]),
            (oa & ob, "lambda c,a: 0", []),
            (oa | ob, "lambda c,a: (~c) | (c & a)", ["a", "c"]),
            (~(oa & ob) | (oa | ob), "lambda c,a: 1", []),
        ]

        for obdd, text, bdd_vars in tests_list:
            self.assertEqual(f"{obdd}", text)
            self.assertEqual(obdd.variables(), set(bdd_vars))

        self.assertEqual(ob.restrict("a", True), 0)
        self.assertEqual(~(oa & ob) | (oa | ob), 1)

        with self.assertRaises(RuntimeError):
            OBDD(self.d, self.ordering)

    def test_obdd_parser(self):
        oa = OBDD(self.a, self.ordering)
        ob = OBDD(self.c, self.ordering)

        for obdd in [oa, ob, ~ob, oa & ob, oa | ob, ~(oa & ob) | (oa | ob)]:
            self.assertEqual(OBDD(f"{obdd.root}", self.ordering), obdd)

        for a, b in [
            ("a & ~a", "0"),
            ("a | ~a", "1"),
            ("~a | c", "~(a & ~c)"),
            ("~~~~a", "a"),
            ("~(~a|~c)", "a&c"),
        ]:
            self.assertEqual(OBDD(a, self.ordering), OBDD(b, self.ordering))

        with self.assertRaises(RuntimeError):
            OBDD("a|~b", self.ordering)

        with self.assertRaises(RuntimeError):
            OBDD("lambda c,a: a|~b")


if __name__ == "__main__":
    unittest.main()
