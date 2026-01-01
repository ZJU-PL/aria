# coding: utf-8
"""
For testing the knowledge compilation engine
"""

from aria.tests import TestCase, main
from aria.bool.knowledge_compiler.dimacs_parser import parse_cnf_string
from aria.bool.knowledge_compiler.dtree import Dtree_Compiler

cnf_foo = """
p cnf 4 4\n
1 2 3 0\n
-2 3 4 0\n
1 -4 0\n
2 3 -4 0
"""


class TestDTree(TestCase):

    def test_dtree(self):
        clausal_form, nvars = parse_cnf_string(cnf_foo, True)
        print(clausal_form)
        dtree_compiler = Dtree_Compiler(clausal_form)
        dtree = dtree_compiler.el2dt([2, 3, 4, 1])
        print(dtree.separators)
        print(dtree.pick_most())
        leaf = dtree.print_info([])
        print(leaf)
        assert dtree.separators == [1, 4]
        assert dtree.atoms == [1, 2, 3, 4]
        assert dtree.left_child.is_leaf()
        assert dtree.is_full_binary()
        for clause in clausal_form:
            assert clause in dtree.clauses


if __name__ == "__main__":
    main()
