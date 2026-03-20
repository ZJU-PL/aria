# coding: utf-8
"""
For testing the knowledge compilation engine
"""

from pysat.formula import CNF

from aria.tests import TestCase, main
from aria.bool.knowledge_compiler.dimacs_parser import parse_cnf_string
from aria.bool.knowledge_compiler.dnnf import DNNF_Compiler
from aria.bool.knowledge_compiler.dtree import Dtree_Compiler
from aria.prob import WMCOptions, compile_wmc, wmc_count

cnf_foo2 = """
p cnf 4 4\n
1 2 3 0\n
-2 3 4 0\n
1 -4 0\n
2 3 -4 0
"""


class TestDNNF(TestCase):

    def test_dnnf(self):
        import copy

        clausal_form, nvars = parse_cnf_string(cnf_foo2, True)
        dt_compiler = Dtree_Compiler(clausal_form.copy())
        dtree = dt_compiler.el2dt([2, 3, 4, 1])
        dnnf_compiler = DNNF_Compiler(dtree)
        dnnf = dnnf_compiler.compile()
        dnnf.reset()

        a = dnnf_compiler.create_trivial_node(5)

        dnnf_smooth = copy.deepcopy(dnnf)
        dnnf_smooth = dnnf_compiler.smooth(dnnf_smooth)
        dnnf_smooth.reset()

        dnnf_conditioning = copy.deepcopy(dnnf)
        dnnf_conditioning = dnnf_compiler.conditioning(dnnf_conditioning, [1, 2])
        dnnf_conditioning.reset()

        dnnf_conditioning.reset()
        # dnnf_simplified = dnnf_compiler.simplify(dnnf_conditioning)

        dnnf_conjoin = copy.deepcopy(dnnf)
        dnnf_conjoin = dnnf_compiler.conjoin(dnnf_conjoin, [1, 2])
        dnnf_conjoin.reset()

        print("Instance is sat or not? ", dnnf_compiler.is_sat(dnnf))

        dnnf_project = copy.deepcopy(dnnf)
        dnnf_project = dnnf_compiler.project(dnnf_project, [1, 2])
        dnnf_project = dnnf_compiler.simplify(dnnf_project)
        dnnf_project.reset()

        print("Computing Min Card ... result = ", dnnf_compiler.m_card(dnnf))

        dnnf_min = copy.deepcopy(dnnf_smooth)
        dnnf_min = dnnf_compiler.minimize(dnnf_min)

        print("Enumerating all models ....")
        models = dnnf_compiler.enumerate_models(dnnf)
        for x in models:
            print(x)

        print("Enumerating all models with smooth version ....")
        models = dnnf_compiler.enumerate_models(dnnf_smooth)
        for x in models:
            print(x)

        assert True

    def test_single_clause_root_compiles(self):
        clausal_form = [[1, 2]]
        dtree = Dtree_Compiler(clausal_form.copy()).el2dt([1, 2])
        self.assertTrue(dtree.is_leaf())

        dnnf = DNNF_Compiler(dtree).compile()
        self.assertIsNotNone(dnnf)
        self.assertEqual(dnnf.type, "O")

    def test_unit_clause_root_compiles(self):
        clausal_form = [[1]]
        dtree = Dtree_Compiler(clausal_form.copy()).el2dt([1])
        self.assertTrue(dtree.is_leaf())

        dnnf = DNNF_Compiler(dtree).compile()
        self.assertIsNotNone(dnnf)
        self.assertEqual(dnnf.type, "L")
        self.assertEqual(dnnf.literal, 1)

    def test_unit_propagation_to_true_leaf(self):
        clausal_form = [[1], [2]]
        dtree = Dtree_Compiler(clausal_form.copy()).el2dt([1, 2])
        dnnf_compiler = DNNF_Compiler(dtree)

        dnnf = dnnf_compiler.compile()
        self.assertIsNotNone(dnnf)
        self.assertTrue(dnnf_compiler.is_sat(dnnf))
        models = dnnf_compiler.enumerate_models(dnnf)
        self.assertEqual(models, [[1, 2]])

    def test_compile_wmc_uses_exact_backend_on_leaf_cases(self):
        cnf = CNF(from_clauses=[[1]])
        weights = {1: 0.4, -1: 0.6}

        compiled = compile_wmc(cnf, weights, WMCOptions(strict_complements=True))
        self.assertEqual(compiled.backend, "wmc-dnnf")
        self.assertEqual(wmc_count(cnf, weights), 0.4)


if __name__ == "__main__":
    main()
