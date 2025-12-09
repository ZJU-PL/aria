# coding: utf-8
"""
For testing the knowledge compilation engine
"""

from aria.tests import TestCase, main
from aria.bool.knowledge_compiler.dimacs_parser import parse_cnf_string
from aria.bool.knowledge_compiler.dnnf import DNNF_Compiler
from aria.bool.knowledge_compiler.dtree import Dtree_Compiler

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

        print('Instance is sat or not? ', dnnf_compiler.is_sat(dnnf))

        dnnf_project = copy.deepcopy(dnnf)
        dnnf_project = dnnf_compiler.project(dnnf_project, [1, 2])
        dnnf_project = dnnf_compiler.simplify(dnnf_project)
        dnnf_project.reset()

        print('Computing Min Card ... result = ', dnnf_compiler.MCard(dnnf))

        dnnf_min = copy.deepcopy(dnnf_smooth)
        dnnf_min = dnnf_compiler.minimize(dnnf_min)

        print('Enumerating all models ....')
        models = dnnf_compiler.enumerate_models(dnnf)
        for x in models:
            print(x)

        print('Enumerating all models with smooth version ....')
        models = dnnf_compiler.enumerate_models(dnnf_smooth)
        for x in models:
            print(x)

        assert True


if __name__ == '__main__':
    main()
